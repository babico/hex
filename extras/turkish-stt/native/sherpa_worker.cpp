// Local, CPU-only worker using Sherpa's stable C ABI for every OS.
#include "sherpa-onnx/c-api/c-api.h"
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
template <typename T, auto Destroy>
using Handle = std::unique_ptr<const T, decltype(Destroy)>;

static std::string file(const fs::path &dir, const char *name) {
    auto path = dir / name;
    if (!fs::is_regular_file(path)) throw std::runtime_error("Missing model file: " + path.string());
    return path.string();
}

class Recognizer {
    Handle<SherpaOnnxOnlineRecognizer, SherpaOnnxDestroyOnlineRecognizer> online{nullptr, SherpaOnnxDestroyOnlineRecognizer};
    Handle<SherpaOnnxOfflineRecognizer, SherpaOnnxDestroyOfflineRecognizer> offline{nullptr, SherpaOnnxDestroyOfflineRecognizer};
    int padding = 16000;
public:
    Recognizer(const std::string &id, fs::path dir, int threads) {
        if (id == "ufakzeka-kulak" && fs::is_directory(dir / "sherpa-onnx-tdt"))
            dir /= "sherpa-onnx-tdt";
        auto encoder = file(dir, "encoder.int8.onnx");
        auto decoder = file(dir, id == "mihu-v2" || id == "ufakzeka-kulak" ? "decoder.int8.onnx" : "decoder.onnx");
        auto joiner = file(dir, "joiner.int8.onnx");
        auto tokens = file(dir, "tokens.txt");
        if (id == "ufakzeka-kulak") {
            SherpaOnnxOfflineRecognizerConfig c{};
            c.feat_config.sample_rate = 16000;
            c.feat_config.feature_dim = 80;
            c.model_config.transducer.encoder = encoder.c_str();
            c.model_config.transducer.decoder = decoder.c_str();
            c.model_config.transducer.joiner = joiner.c_str();
            c.model_config.tokens = tokens.c_str();
            c.model_config.model_type = "nemo_transducer";
            c.model_config.num_threads = threads;
            c.model_config.provider = "cpu";
            c.decoding_method = "greedy_search";
            offline.reset(SherpaOnnxCreateOfflineRecognizer(&c));
            if (!offline) throw std::runtime_error("Could not load offline Kulak model");
        } else if (id == "mihu-v2" || id == "seda-v2" || id == "seda-tiny") {
            SherpaOnnxOnlineRecognizerConfig c{};
            c.feat_config.sample_rate = 16000;
            c.feat_config.feature_dim = 80;
            c.model_config.transducer.encoder = encoder.c_str();
            c.model_config.transducer.decoder = decoder.c_str();
            c.model_config.transducer.joiner = joiner.c_str();
            c.model_config.tokens = tokens.c_str();
            c.model_config.num_threads = threads;
            c.model_config.provider = "cpu";
            c.decoding_method = "modified_beam_search";
            c.max_active_paths = 4;
            c.enable_endpoint = 0;
            online.reset(SherpaOnnxCreateOnlineRecognizer(&c));
            if (!online) throw std::runtime_error("Could not load streaming transducer");
            padding = id == "mihu-v2" ? 8000 : 16000;
        } else {
            throw std::runtime_error("Unsupported model: " + id);
        }
    }
    std::string decode(const std::string &path) {
        Handle<SherpaOnnxWave, SherpaOnnxFreeWave> wave(SherpaOnnxReadWave(path.c_str()), SherpaOnnxFreeWave);
        if (!wave || wave->sample_rate != 16000 || wave->num_samples <= 0 || wave->num_samples > 9600000)
            throw std::runtime_error("Expected non-empty mono 16 kHz PCM16 WAV, max 10 minutes");
        if (offline) {
            // The author validates Sherpa 1.13.8 against NeMo at input gain 8.
            std::vector<float> samples(wave->samples, wave->samples + wave->num_samples);
            for (auto &sample : samples) sample *= 8.0f;
            Handle<SherpaOnnxOfflineStream, SherpaOnnxDestroyOfflineStream> s(SherpaOnnxCreateOfflineStream(offline.get()), SherpaOnnxDestroyOfflineStream);
            if (!s) throw std::runtime_error("Could not create offline stream");
            SherpaOnnxAcceptWaveformOffline(s.get(), 16000, samples.data(), static_cast<int32_t>(samples.size()));
            SherpaOnnxDecodeOfflineStream(offline.get(), s.get());
            Handle<char, SherpaOnnxDestroyOfflineStreamResultJson> json(SherpaOnnxGetOfflineStreamResultAsJson(s.get()), SherpaOnnxDestroyOfflineStreamResultJson);
            if (!json) throw std::runtime_error("Missing offline result");
            return json.get();
        }
        Handle<SherpaOnnxOnlineStream, SherpaOnnxDestroyOnlineStream> s(SherpaOnnxCreateOnlineStream(online.get()), SherpaOnnxDestroyOnlineStream);
        if (!s) throw std::runtime_error("Could not create online stream");
        SherpaOnnxOnlineStreamAcceptWaveform(s.get(), 16000, wave->samples, wave->num_samples);
        const std::vector<float> silence(padding, 0.0f);
        SherpaOnnxOnlineStreamAcceptWaveform(s.get(), 16000, silence.data(), padding);
        SherpaOnnxOnlineStreamInputFinished(s.get());
        while (SherpaOnnxIsOnlineStreamReady(online.get(), s.get())) SherpaOnnxDecodeOnlineStream(online.get(), s.get());
        Handle<char, SherpaOnnxDestroyOnlineStreamResultJson> json(SherpaOnnxGetOnlineStreamResultAsJson(online.get(), s.get()), SherpaOnnxDestroyOnlineStreamResultJson);
        if (!json) throw std::runtime_error("Missing online result");
        return json.get();
    }
};

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "hex-sherpa-worker --weights DIR --model ID --language tr [--threads N] [--wav FILE | --serve 1]\n";
            std::cout << "Sherpa-ONNX " << SherpaOnnxGetVersionStr() << '\n';
            return 0;
        }
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            if (i + 1 >= argc) throw std::runtime_error("Expected option and value");
            std::string k = argv[i];
            if (k != "--weights" && k != "--model" && k != "--language" && k != "--threads" && k != "--wav" && k != "--serve")
                throw std::runtime_error("Unknown option: " + k);
            if (!args.emplace(k, argv[i+1]).second) throw std::runtime_error("Duplicate option: " + k);
        }
        if (args.at("--language") != "tr") throw std::runtime_error("This worker supports Turkish");
        int threads = args.count("--threads") ? std::stoi(args.at("--threads")) : 4;
        if (threads < 1 || threads > 64) throw std::runtime_error("Expected 1-64 threads");
        const bool serve = args.count("--serve") && args.at("--serve") == "1";
        if (serve == bool(args.count("--wav"))) throw std::runtime_error("Choose --wav or --serve 1");
        Recognizer recognizer(args.at("--model"), fs::path(args.at("--weights")), threads);
        if (serve) {
            // One input path per line; one JSON result per line. EOF destroys weights.
            // The process is local, and stdout is reserved for this protocol.
            std::cout << "{\"ready\":true}" << std::endl;
            std::string path;
            while (std::getline(std::cin, path)) {
                if (path.size() > 32768) throw std::runtime_error("Invalid path");
                auto json = recognizer.decode(path);
                // Sherpa JSON may be pretty-printed. Strip line breaks (escaped
                // line breaks within quoted values are unaffected).
                for (auto &ch : json) if (ch == '\n' || ch == '\r') ch = ' ';
                std::cout << json << std::endl;
            }
        } else {
            std::cout << recognizer.decode(args.at("--wav")) << std::endl;
        }
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
