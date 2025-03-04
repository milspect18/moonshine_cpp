/**
 * @file moonshine_transcribe.cpp
 * @brief Convenience class to use tokenizers_cpp with Moonshine onnx tokens vector output.
 */

#include <fstream>
#include <sstream>
#include <stdexcept>
#include "moonshine.h"


namespace Moonshine {

Transcriber::Transcriber(const ModelType model_type,
                         const f_path &encoder_path,
                         const f_path &decoder_path,
                         const f_path &tokenizer_path,
                         const int num_threads)
{
    if (!std::filesystem::exists(tokenizer_path)) {
        throw std::runtime_error("File not found: " + tokenizer_path.string());
    } else if (!std::filesystem::is_regular_file(tokenizer_path)) {
        throw std::runtime_error("Not a regular file: " + tokenizer_path.string());
    }

    std::ifstream ifs(tokenizer_path);
    std::stringstream buffer;

    buffer << ifs.rdbuf();

    tokenizer = tokenizers::Tokenizer::FromBlobJSON(buffer.str());

    switch (model_type) {
        case ModelType::Base:
            model = std::make_unique<OnnxModel>(OnnxModel::Base(encoder_path, decoder_path, num_threads));
            break;
        case ModelType::Tiny:
            model = std::make_unique<OnnxModel>(OnnxModel::Tiny(encoder_path, decoder_path, num_threads));
            break;
    }
}

std::string Transcriber::transcribe(const std::vector<float> &audio_data) noexcept {
    auto tokens = model->run(const_cast<std::vector<float> &>(audio_data));

    if (tokens.empty()) {
        return "";
    }

    return tokenizer->Decode(tokens);
}

std::string Transcriber::operator()(const std::vector<float> &audio_data) noexcept {
    return transcribe(audio_data);
}

} // namespace Moonshine