/** Moonshine Onnx implementation
 *
 * This class is a wrapper around the ONNX Runtime C++ API. It provides a simple
 * interface to load and run ONNX models. The class has two static methods to
 * create instances of the class with different model configurations. The run
 * method takes a vector of floats as input and returns a vector of integers as
 * output.  The class is designed to be used with the Moonshine tokenizer.
 */

#include <stdexcept>
#include <cmath>
#include "moonshine.h"

namespace {
    /**
     * @brief Clones an existing tensor.
     *
     * The Ort::Value tensors are treated similar to unique_pointers in that they are move only.
     * This presents a challenge for the decoder where the auto-regressive nature of the model
     * requires that previous iteration tensor values be passed in future iterations (i.e. the
     * key-value cache and encoder hidden states). This function allows for the cloning of these
     * tensors to ensure that the original values are not modified.
     *
     * @tparam T The data type of the tensor elements.
     * @param tensor The tensor to be cloned.
     * @param mem_info The memory information for the new tensor.
     * @return Ort::Value A new tensor that is a clone of the input tensor.
     */
    template<typename T>
    Ort::Value clone_tensor(Ort::Value &tensor, Ort::MemoryInfo &mem_info) {
        return Ort::Value::CreateTensor<T>(
            mem_info,
            tensor.GetTensorMutableData<T>(),
            tensor.GetTensorTypeAndShapeInfo().GetElementCount(),
            tensor.GetTensorTypeAndShapeInfo().GetShape().data(),
            tensor.GetTensorTypeAndShapeInfo().GetShape().size()
        );
    }
}

namespace Moonshine {

OnnxModel::OnnxModel(const f_path &encoder_path,
                     const f_path &decoder_path,
                     const int64_t num_layers,
                     const int64_t num_kv_heads,
                     const int64_t head_dim,
                     const int num_threads,
                     Ort::Env env,
                     Ort::MemoryInfo memory_info)
    : num_layers(num_layers),
      num_kv_heads(num_kv_heads),
      head_dim(head_dim),
      env(std::move(env)),
      memory_info(std::move(memory_info)),
      encoder(nullptr),
      decoder(nullptr)
{
    if (!std::filesystem::is_regular_file(encoder_path)) {
        throw std::runtime_error("Encoder path is not a regular file: " + encoder_path.string());
    }

    if (!std::filesystem::is_regular_file(decoder_path)) {
        throw std::runtime_error("Decoder path is not a regular file: " + decoder_path.string());
    }

    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(num_threads);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.DisableCpuMemArena();

    encoder = Ort::Session(this->env, encoder_path.c_str(), options);
    decoder = Ort::Session(this->env, decoder_path.c_str(), options);

    initialize_model_io_names();
}

OnnxModel OnnxModel::Base(const f_path &encoder_path,
                          const f_path &decoder_path,
                          const int num_threads)
{
    return OnnxModel(encoder_path, decoder_path, 8, 8, 52, num_threads);
}

OnnxModel OnnxModel::Tiny(const f_path &encoder_path,
                          const f_path &decoder_path,
                          const int num_threads)
{
    return OnnxModel(encoder_path, decoder_path, 6, 8, 36, num_threads);
}

void OnnxModel::initialize_model_io_names() {
    for (size_t i = 0; i < encoder.GetInputCount(); i++) {
        auto input_name = encoder.GetInputNameAllocated(i, model_name_allocator);
        encoder_input_names.push_back(input_name.get());
        model_io_names.push_back(std::move(input_name));
    }

    for (size_t i = 0; i < encoder.GetOutputCount(); i++) {
        auto output_name = encoder.GetOutputNameAllocated(i, model_name_allocator);
        encoder_output_names.push_back(output_name.get());
        model_io_names.push_back(std::move(output_name));
    }

    for (size_t i = 0; i < decoder.GetInputCount(); i++) {
        auto input_name = decoder.GetInputNameAllocated(i, model_name_allocator);
        decoder_input_names.push_back(input_name.get());
        model_io_names.push_back(std::move(input_name));
    }

    for (size_t i = 0; i < decoder.GetOutputCount(); i++) {
        auto output_name = decoder.GetOutputNameAllocated(i, model_name_allocator);
        decoder_output_names.push_back(output_name.get());
        model_io_names.push_back(std::move(output_name));
    }
}

std::vector<int> OnnxModel::run(std::vector<float> &audio_data) noexcept {
    double audio_len = static_cast<double>(audio_data.size()) / sample_rate;
    size_t max_len = std::round(audio_len * max_tokens_per_second);

    auto last_hidden_state = encode(audio_data);
    return decode(std::move(last_hidden_state.at(0)), max_len);
}

std::vector<Ort::Value> OnnxModel::encode(std::vector<float> &audio_data) {
    std::vector<int64_t> encoder_input_shape = {1, static_cast<int64_t>(audio_data.size())};

    auto in_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        audio_data.data(),
        audio_data.size(),
        encoder_input_shape.data(),
        encoder_input_shape.size()
    );

    return encoder.Run(
        Ort::RunOptions{nullptr},
        encoder_input_names.data(),
        &in_tensor,
        1,
        encoder_output_names.data(),
        encoder_output_names.size()
    );
}

std::vector<int> OnnxModel::decode(Ort::Value last_hidden_state, size_t max_len) {
    size_t max_token_count = std::max(max_len, min_token_count); // Ensure at least one token is generated
    auto past_key_values = initialize_past_key_values();
    std::vector<int> result_tokens{};
    std::vector<int64_t> cur_tokens{ start_token };

    for (size_t i = 0; i < max_token_count; i++) {
        bool use_cache_branch = i > 0;

        auto output = decode_next_token(
            cur_tokens,
            last_hidden_state,
            past_key_values,
            use_cache_branch
        );

        auto next_token = get_next_token(std::move(output.at(0)));

        cur_tokens.clear();
        cur_tokens.push_back(next_token);

        if (next_token == end_token) {
            break;
        }

        result_tokens.push_back(static_cast<int>(next_token));

        std::vector<Ort::Value> present_kv;

        for (auto out_iter = output.begin() + 1; out_iter != output.end(); ++out_iter) {
            present_kv.emplace_back(std::move(*out_iter));
        }

        update_kv_cache(past_key_values, present_kv, use_cache_branch);
    }

    return result_tokens;
}

std::vector<Ort::Value> OnnxModel::decode_next_token(std::vector<int64_t> &cur_tokens,
                                                     Ort::Value &last_hidden_state,
                                                     std::vector<Ort::Value> &past_key_values,
                                                     bool use_cache_branch)
{
    std::vector<Ort::Value> decoder_inputs;

    std::array<int64_t, 2> dec_input_ids_shape{1, static_cast<int64_t>(cur_tokens.size())};
    decoder_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>( // input_ids
        memory_info,
        cur_tokens.data(),
        cur_tokens.size(),
        dec_input_ids_shape.data(),
        dec_input_ids_shape.size()
    ));

    decoder_inputs.emplace_back(clone_tensor<float>(last_hidden_state, memory_info));

    for (auto &v : past_key_values) {
        decoder_inputs.emplace_back(clone_tensor<float>(v, memory_info));
    }

    std::array<int64_t, 1> dec_use_cache_branch_shape{1};
    std::array<bool, 1> dec_use_cache_branch_data{use_cache_branch};
    decoder_inputs.emplace_back(Ort::Value::CreateTensor<bool>(
        memory_info,
        dec_use_cache_branch_data.data(),
        dec_use_cache_branch_data.size(),
        dec_use_cache_branch_shape.data(),
        dec_use_cache_branch_shape.size()
    ));

    return decoder.Run(
        Ort::RunOptions{nullptr},
        decoder_input_names.data(),
        decoder_inputs.data(),
        decoder_inputs.size(),
        decoder_output_names.data(),
        decoder_output_names.size()
    );
}

std::vector<Ort::Value> OnnxModel::initialize_past_key_values() {
    std::vector<Ort::Value> past_key_values;
    std::array<int64_t, 4> shape = {0, num_kv_heads, 1, head_dim};

    for (const auto &key : decoder_input_names) {
        const std::string key_str(key);
        const std::string past_key_prefix("past_key_values");

        if (key_str.find(past_key_prefix) != std::string::npos) {
            past_key_values.emplace_back(Ort::Value::CreateTensor<float>(
                memory_info,
                nullptr,  // No data needed since first dimension is 0
                0,        // Total element count is 0
                shape.data(),
                shape.size()
            ));
        }
    }

    return past_key_values;
}

void OnnxModel::update_kv_cache(std::vector<Ort::Value> &cache,
                                std::vector<Ort::Value> &new_values,
                                bool use_cache_branch)
{
    auto cache_iter = cache.begin();
    auto new_iter = new_values.begin();

    for (; cache_iter != cache.end() && new_iter != new_values.end(); ++cache_iter, ++new_iter) {
        auto present_kv_shape = new_iter->GetTensorTypeAndShapeInfo().GetShape();
        auto past_kv_shape = cache_iter->GetTensorTypeAndShapeInfo().GetShape();

        bool past_kv_empty = (past_kv_shape[0] == 0);
        int64_t present_kv_tok_proc = present_kv_shape[2];
        bool is_decoder_kv = present_kv_tok_proc > 1;

        if (past_kv_empty || is_decoder_kv || !use_cache_branch) {
            *cache_iter = std::move(*new_iter);
        }
    }
}

int OnnxModel::get_next_token(Ort::Value logits) {
    auto shape = logits.GetTensorTypeAndShapeInfo().GetShape();

    // Validate the shape is as expected [1,1,vocabulary_size]
    if (shape.size() != 3 || shape[0] != 1 || shape[1] != 1) {
        throw std::runtime_error("Unexpected logits shape");
    }

    int64_t vocab_size = shape[2];
    const float *p_logit_data = logits.GetTensorData<float>();
    std::vector<float> logit_vec(p_logit_data, p_logit_data + vocab_size);

    // Find the index of the maximum value in the logits
    return std::distance(logit_vec.begin(), std::max_element(logit_vec.begin(), logit_vec.end()));
}

} // namespace Moonshine