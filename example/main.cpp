#include "moonshine.h"
#include "AudioFile.h"

void transcribe_all(Moonshine::Transcriber& stt, const std::string& in_path);
std::vector<std::filesystem::path> get_wav_paths(const std::string& in_path);
std::vector<float> read_wav_file(const std::string& path);


int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <model> <encoder.onnx> <decoder.onnx> <tok.json> <wav_f_path>" << std::endl;

        return 1;
    }

    auto model_type = Moonshine::ModelType::from_string(argv[1]);

    if (!model_type) {
        std::cerr << "Invalid model name. Use 'base' or 'tiny'." << std::endl;
        return 1;
    }

    auto stt = Moonshine::Transcriber(*model_type, argv[2], argv[3], argv[4]);

    transcribe_all(stt, argv[5]);

    return 0;
}


void transcribe_all(Moonshine::Transcriber& stt, const std::string& in_path) {
    auto wav_files = get_wav_paths(in_path);

    std::cout << "wav_name,length,rtf,transcript\n";

    for (const auto& wav_file : wav_files) {
        auto audio_data = read_wav_file(wav_file);
        double length = static_cast<double>(audio_data.size()) / 16000.0;

        if (audio_data.empty()) {
            std::cerr << "\"" << wav_file.stem() << "\",Invalid,audio,file\n";

            continue;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        auto transcript = stt.transcribe(audio_data);

        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        double rtf = elapsed.count() / length;

        std::cout << "\"" << wav_file.stem()
                  << "\"," << length
                  << "," << rtf
                  << ",\"\"\"" << transcript << "\"\"\"\n";
    }
}

std::vector<std::filesystem::path> get_wav_paths(const std::string& in_path) {
    std::filesystem::path p(in_path);

    if (!std::filesystem::exists(p)) {
        std::cerr << "File not found: " << p << std::endl;
        std::exit(1);
    }

    std::vector<std::filesystem::path> wav_files;

    if (std::filesystem::is_regular_file(p)) {
        wav_files.push_back(p);
    } else if (std::filesystem::is_directory(p)) {
        for (const auto& entry : std::filesystem::directory_iterator(p)) {
            if (entry.is_regular_file() && entry.path().extension() == ".wav") {
                wav_files.push_back(entry.path());
            }
        }
    } else {
        std::cerr << "Not a regular file or directory: " << p << std::endl;
        std::exit(1);
    }

    return wav_files;
}

std::vector<float> read_wav_file(const std::string& path) {
    std::filesystem::path p(path);

    if (!std::filesystem::exists(p)) {
        std::cerr << "File not found: " << p << std::endl;
        std::exit(1);
    } else if (!std::filesystem::is_regular_file(p)) {
        std::cerr << "Not a regular file: " << p << std::endl;
        std::exit(1);
    }

    AudioFile<double> af;
    af.load(p.string());

    if (af.getNumChannels() != 1 || af.getBitDepth() != 16 || af.getSampleRate() != 16000) {
        return {};
    }

    std::vector<float> data;
    data.reserve(af.getNumSamplesPerChannel());

    for (size_t i = 0; i < af.getNumSamplesPerChannel(); ++i) {
        data.push_back(af.samples[0][i]);
    }

    return data;
}