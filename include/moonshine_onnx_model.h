#ifndef MOONSHINE_ONNX_MODEL_H__
#define MOONSHINE_ONNX_MODEL_H__

/**
 * @file moonshine_onnx_model.h
 * @brief ONNX model inference implementation for the Moonshine model
 */

#include <filesystem>
#include <optional>
#include "onnxruntime_cxx_api.h"


namespace {
    /**
     * @brief Creates a default ONNX runtime environment
     * @return Ort::Env Configured ONNX runtime environment
     */
    inline Ort::Env default_env() {
        return Ort::Env{ORT_LOGGING_LEVEL_WARNING, "Moonshine::OnnxModel"};
    }

    /**
     * @brief Creates default memory information for ONNX runtime
     * @return Ort::MemoryInfo Memory information for CPU-based allocation
     */
    inline Ort::MemoryInfo default_memory_info() {
        return Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator,
                                          OrtMemType::OrtMemTypeCPU);
    }
}


namespace Moonshine {

using f_path = std::filesystem::path;

/**
 * @class OnnxModel
 * @brief Encapsulates the speech recognition model using ONNX Runtime
 *
 * This class provides functionality for loading and running encoder-decoder
 * speech recognition models via ONNX Runtime. It handles both the encoding
 * of audio data and the decoding of features into token ids.
 */
class OnnxModel {
public:
    /**
     * @brief Creates a Base model instance
     *
     * @param encoder_path Path to the encoder ONNX model file
     * @param decoder_path Path to the decoder ONNX model file
     * @param num_threads Number of threads to use for inference (default: 4)
     * @return OnnxModel Configured Base model instance
     */
    static OnnxModel Base(const f_path &encoder_path,
                          const f_path &decoder_path,
                          const int num_threads = 4);

    /**
     * @brief Creates a Tiny model instance
     *
     * @param encoder_path Path to the encoder ONNX model file
     * @param decoder_path Path to the decoder ONNX model file
     * @param num_threads Number of threads to use for inference (default: 4)
     * @return OnnxModel Configured Tiny model instance
     */
    static OnnxModel Tiny(const f_path &encoder_path,
                          const f_path &decoder_path,
                          const int num_threads = 4);

    /**
     * @brief Runs inference on audio data to produce token indices
     *
     * This method encodes the audio data and then decodes the features
     * to produce a sequence of token indices representing the transcription.
     *
     * @param audio_data Vector of float audio samples (assumed to be 16kHz mono)
     * @return std::vector<int> Vector of token indices representing the transcription
     */
    std::vector<int> run(std::vector<float> &audio_data) noexcept;

    /**
     * @brief Gets the required sample rate for the model
     * @return size_t Sample rate in Hz (16000)
     */
    constexpr inline static size_t get_sample_rate() noexcept { return sample_rate; }

private:
    /**
     * @brief Constructs a new OnnxModel instance
     *
     * @param encoder_path Path to the encoder ONNX model file
     * @param decoder_path Path to the decoder ONNX model file
     * @param num_layers Number of layers in the model
     * @param num_kv_heads Number of key-value heads in the model
     * @param head_dim Dimension of each attention head
     * @param num_threads Number of threads to use for inference
     * @param env ONNX runtime environment (default: Ort::Env{ORT_LOGGING_LEVEL_WARNING, "Moonshine::OnnxModel"})
     * @param memory_info Memory information (default: Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPU))
     */
    OnnxModel(const f_path &encoder_path,
              const f_path &decoder_path,
              const int64_t num_layers,
              const int64_t num_kv_heads,
              const int64_t head_dim,
              const int num_threads,
              Ort::Env env = default_env(),
              Ort::MemoryInfo memory_info = default_memory_info());

    /**
     * @brief Initializes input/output names for the models
     */
    void initialize_model_io_names();

    /**
     * @brief Encodes audio data into latent space representations
     *
     * @param audio_data Vector of float audio samples
     * @return std::vector<Ort::Value> Encoder output tensors
     */
    std::vector<Ort::Value> encode(std::vector<float> &audio_data);

    /**
     * @brief Decodes encoder output into token indices
     *
     * @param last_hidden_state Encoder's hidden state output
     * @param max_len Maximum length of the output sequence
     * @return std::vector<int> Vector of decoded token indices
     */
    std::vector<int> decode(Ort::Value last_hidden_state, size_t max_len);

    /**
     * @brief Generate a zero filled key-value cache for decoding
     * @return std::vector<Ort::Value> Initialized past key-value tensors
     */
    std::vector<Ort::Value> initialize_past_key_values();

    /**
     * @brief Gets the next token from logits
     *
     * This method currently returns the index of the maximum value in the logits tensor.
     * A more sophisticated method could be used to sample from the distribution.
     *
     * @param logits Logits tensor from the decoder
     * @return int The index of the next token
     */
    int get_next_token(Ort::Value logits);

    /**
     * @brief Performs one decoding step to get the next token
     *
     * @param cur_tokens Current sequence of tokens
     * @param last_hidden_state Current hidden state
     * @param past_key_values Current key-value cache
     * @param use_cache_branch Whether to use the caching branch of the model
     * @return std::vector<Ort::Value> Decoder outputs including new logits and key-value cache
     */
    std::vector<Ort::Value> decode_next_token(std::vector<int64_t> &cur_tokens,
                                              Ort::Value &last_hidden_state,
                                              std::vector<Ort::Value> &past_key_values,
                                              bool use_cache_branch);

    /**
     * @brief Updates the key-value cache with new values
     *
     * @param cache Current key-value cache to update
     * @param new_values New values to add to the cache
     * @param use_cache_branch Whether the cache branch of the model is being used
     */
    void update_kv_cache(std::vector<Ort::Value> &cache,
                         std::vector<Ort::Value> &new_values,
                         bool use_cache_branch);

    int64_t num_layers;
    int64_t num_kv_heads;
    int64_t head_dim;

    Ort::Env env;         /**< ONNX runtime environment */
    Ort::MemoryInfo memory_info;    /**< Memory allocation information */
    Ort::Session encoder; /**< ONNX runtime session for the encoder */
    Ort::Session decoder; /**< ONNX runtime session for the decoder */

    Ort::AllocatorWithDefaultOptions model_name_allocator; /**< Allocator for model I/O names */
    std::vector<Ort::AllocatedStringPtr> model_io_names;   /**< Storage for allocated string pointers */

    std::vector<const char *> encoder_input_names;  /**< Input names for the encoder */
    std::vector<const char *> encoder_output_names; /**< Output names for the encoder */

    std::vector<const char *> decoder_input_names;  /**< Input names for the decoder */
    std::vector<const char *> decoder_output_names; /**< Output names for the decoder */

    static constexpr int start_token = 1;           /**< Token ID representing sequence start */
    static constexpr int end_token = 2;             /**< Token ID representing sequence end */
    static constexpr size_t sample_rate = 16000;    /**< Expected audio sample rate in Hz */
    static constexpr size_t max_tokens_per_second = 6;  /**< Maximum tokens per second of audio */
    static constexpr size_t min_token_count = 1;    /**< Minimum number of tokens to generate */
};

}

#endif
