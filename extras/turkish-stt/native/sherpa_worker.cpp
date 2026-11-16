// Local CPU inference via Sherpa-ONNX 1.13.8. No Python or network.
#include "sherpa-onnx/c-api/cxx-api.h"
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace sherpa_onnx::cxx;

static std::string file(const fs::path &dir, const char *name) {
    const auto path = dir / name;
    if (!fs::is_regular_file(path)) throw std::runtime_error("Missing model file: " + path.string());
    return path.string();
}
int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "hex-sherpa-worker --weights DIRECTORY --model mihu-v2|seda-v2|seda-tiny|ufakzeka-kulak --language tr --wav FILE [--threads N]\n";
            return 0;
        }
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::runtime_error("Expected option and value");
            const std::string key = argv[i];
            if (key != "--weights" && key != "--model" && key != "--language" &&
                key != "--wav" && key != "--threads") throw std::runtime_error("Unknown option: " + key);
            if (!args.emplace(key, argv[i+1]).second) throw std::runtime_error("Duplicate option: " + key);
        }
        const std::string id = args.at("--model");
        const fs::path weights = args.at("--weights");
        if (args.at("--language") != "tr") throw std::runtime_error("This worker supports Turkish");
        const int threads = args.count("--threads") ? std::stoi(args.at("--threads")) : 4;
        if (threads < 1 || threads > 64) throw std::runtime_error("Expected 1-64 threads");
        auto wave = ReadWave(args.at("--wav"));
        if (wave.sample_rate != 16000 || wave.samples.empty()) throw std::runtime_error("Expected non-empty mono 16 kHz WAV");
        if (id == "ufakzeka-kulak") {
            // Published parity settings for 1.13.8: gain 8, no CTC padding for TDT.
            const fs::path dir = fs::is_directory(weights / "sherpa-onnx-tdt") ? weights / "sherpa-onnx-tdt" : weights;
            OfflineRecognizerConfig config;
            config.model_config.transducer.encoder = file(dir, "encoder.int8.onnx");
            config.model_config.transducer.decoder = file(dir, "decoder.int8.onnx");
            config.model_config.transducer.joiner = file(dir, "joiner.int8.onnx");
            config.model_config.tokens = file(dir, "tokens.txt");
            config.model_config.model_type = "nemo_transducer";
            config.model_config.num_threads = threads;
            config.model_config.provider = "cpu";
            config.decoding_method = "greedy_search";
            const auto recognizer = OfflineRecognizer::Create(config);
            if (!recognizer.Get()) throw std::runtime_error("Could not load Kulak TDT");
            for (auto &sample : wave.samples) sample *= 8.0f;
            auto stream = recognizer.CreateStream();
            stream.AcceptWaveform(16000, wave.samples.data(), static_cast<int32_t>(wave.samples.size()));
            recognizer.Decode(&stream);
            std::cout << recognizer.GetResult(&stream).json << '\n';
        } else if (id == "mihu-v2" || id == "seda-v2" || id == "seda-tiny") {
            OnlineRecognizerConfig config;
            config.model_config.transducer.encoder = file(weights, "encoder.int8.onnx");
            config.model_config.transducer.decoder = file(weights, id == "mihu-v2" ? "decoder.int8.onnx" : "decoder.onnx");
            config.model_config.transducer.joiner = file(weights, "joiner.int8.onnx");
            config.model_config.tokens = file(weights, "tokens.txt");
            config.model_config.num_threads = threads;
            config.model_config.provider = "cpu";
            config.decoding_method = "modified_beam_search";
            config.max_active_paths = 4;
            config.enable_endpoint = false;
            const auto recognizer = OnlineRecognizer::Create(config);
            if (!recognizer.Get()) throw std::runtime_error("Could not load online transducer");
            auto stream = recognizer.CreateStream();
            stream.AcceptWaveform(16000, wave.samples.data(), static_cast<int32_t>(wave.samples.size()));
            const std::vector<float> silence(id == "mihu-v2" ? 8000 : 16000, 0.0f);
            stream.AcceptWaveform(16000, silence.data(), static_cast<int32_t>(silence.size()));
            stream.InputFinished();
            while (recognizer.IsReady(&stream)) recognizer.Decode(&stream);
            std::cout << recognizer.GetResult(&stream).json << '\n';
        } else {
            throw std::runtime_error("Unsupported native Sherpa model: " + id);
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
