#ifndef MOONSHINE_H__
#define MOONSHINE_H__

/**
 * @file moonshine.h
 * @brief Main header file for the moonshine_cpp speech recognition library
 */

#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include "moonshine_onnx_model.h"
#include "tokenizers_cpp.h"


namespace Moonshine {

/**
 * @class ModelType
 * @brief Represents the available speech recognition model types
 *
 * This class provides a type-safe enumeration for the different speech
 * recognition models available in the Moonshine library.
 */
class ModelType {
public:
    /**
     * @enum Type
     * @brief Enumeration of supported model types
     */
    enum Type: uint8_t {
        Base,  /** Standard accuracy model */
        Tiny   /** Smaller, faster model with reduced accuracy */
    };

    /**
     * @brief Construct a new Model Type object
     * @param type The model type to create
     */
    constexpr ModelType(Type type) noexcept : type(type) {}

    /**
     * @brief Create a ModelType from a string representation
     * @param str String representation of the model type ("base" or "tiny")
     * @return std::optional<ModelType> The corresponding ModelType or std::nullopt if invalid
     */
    static std::optional<ModelType> from_string(const std::string &str) noexcept {
        std::string str_lower(str);
        std::transform(str_lower.begin(), str_lower.end(), str_lower.begin(), ::tolower);

        if (str_lower == "base") {
            return ModelType(Base);
        } else if (str_lower == "tiny") {
            return ModelType(Tiny);
        }

        return std::nullopt;
    }

    constexpr inline bool operator==(const ModelType &other) const noexcept {
        return type == other.type;
    }

    constexpr inline bool operator!=(const ModelType &other) const noexcept {
        return type != other.type;
    }

    constexpr inline bool operator==(const Type &other) const noexcept {
        return type == other;
    }

    constexpr inline bool operator!=(const Type &other) const noexcept {
        return type != other;
    }

    /**
     * @brief Implicit conversion to Type enum
     * @return The underlying Type enum value
     */
    constexpr inline operator Type() const noexcept {
        return type;
    }

private:
    Type type;  /**< The underlying model type value */
};

/**
 * @class Transcriber
 * @brief Main class for performing speech-to-text transcription
 *
 * This class encapsulates the speech recognition engine and provides methods
 * to transcribe audio data into text.
 */
class Transcriber {
public:
    /**
     * @brief Construct a new Transcriber object
     *
     * @param model_type The type of model to use (Base or Tiny)
     * @param encoder_path Path to the encoder ONNX model file
     * @param decoder_path Path to the decoder ONNX model file
     * @param tokenizer_path Path to the tokenizer model (JSON) file
     * @param num_threads Number of threads to use for inference (default: 4)
     */
    Transcriber(const ModelType model_type,
                const f_path &encoder_path,
                const f_path &decoder_path,
                const f_path &tokenizer_path,
                const int num_threads = 4);

    /**
     * @brief Transcribe audio data to text
     *
     * @param audio_data Vector of float audio samples (assumed to be 16kHz mono)
     * @return std::string The transcribed text
     */
    std::string transcribe(const std::vector<float> &audio_data) noexcept;

    /**
     * @brief Operator overload for convenient function-call syntax
     *
     * This is a convenience wrapper around the transcribe() method.
     *
     * @param audio_data Vector of float audio samples (assumed to be 16kHz mono)
     * @return std::string The transcribed text
     */
    std::string operator()(const std::vector<float> &audio_data) noexcept;

private:
    std::unique_ptr<OnnxModel> model;       /**< The ONNX model for inference */
    std::unique_ptr<tokenizers::Tokenizer> tokenizer;  /**< The tokenizer for text processing */
};

}

#endif
