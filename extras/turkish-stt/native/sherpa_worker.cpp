// Local, CPU-only worker using Sherpa's stable C ABI for every OS.
#include "sherpa-onnx/c-api/c-api.h"
#include <cstdint>
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
    if (!fs::is_regular_file(path)) throw std::runtime_error("Missing model file: " + path.u8string());
    return path.u8string();
}

static std::vector<float> parse_pcm16(const std::string &hex) {
    if (hex.empty() || hex.size() > 131072 || hex.size() % 4 != 0)
        throw std::runtime_error("Invalid PCM16 chunk");
    auto digit = [](char c)->unsigned {
        if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return static_cast<unsigned>(c - 'A' + 10);
        throw std::runtime_error("Invalid PCM16 encoding");
    };
    std::vector<float> samples;
    samples.reserve(hex.size() / 4);
    for (size_t i = 0; i < hex.size(); i += 4) {
        const uint16_t lo = static_cast<uint16_t>((digit(hex[i]) << 4) | digit(hex[i+1]));
        const uint16_t hi = static_cast<uint16_t>((digit(hex[i+2]) << 4) | digit(hex[i+3]));
        const uint16_t bits = static_cast<uint16_t>(lo | (hi << 8));
        const int16_t signed_sample = static_cast<int16_t>(bits);
        samples.push_back(static_cast<float>(signed_sample) / 32768.0f);
    }
    return samples;
}

class Recognizer {
    Handle<SherpaOnnxOnlineRecognizer, SherpaOnnxDestroyOnlineRecognizer> online{nullptr, SherpaOnnxDestroyOnlineRecognizer};
    Handle<SherpaOnnxOfflineRecognizer, SherpaOnnxDestroyOfflineRecognizer> offline{nullptr, SherpaOnnxDestroyOfflineRecognizer};
    Handle<SherpaOnnxOnlineStream, SherpaOnnxDestroyOnlineStream> active{nullptr, SherpaOnnxDestroyOnlineStream};
    int padding = 16000;
    bool collecting = false;
    std::vector<float> clip;
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
    bool can_stream() const { return static_cast<bool>(online); }

    std::string begin_stream() {
        if (!online || collecting) throw std::runtime_error("The model cannot begin online partials while another clip is active");
        active.reset(SherpaOnnxCreateOnlineStream(online.get()));
        if (!active) throw std::runtime_error("Could not create live stream");
        return "{\"text\":\"\"}";
    }
    std::string partial_result() {
        if (!active) throw std::runtime_error("No live stream is active");
        Handle<char, SherpaOnnxDestroyOnlineStreamResultJson> json(
            SherpaOnnxGetOnlineStreamResultAsJson(online.get(), active.get()),
            SherpaOnnxDestroyOnlineStreamResultJson);
        if (!json) throw std::runtime_error("Missing streaming result");
        return json.get();
    }
    std::string feed_stream(const std::string &hex) {
        if (!active || hex.empty() || hex.size() > 131072 || hex.size() % 4 != 0)
            throw std::runtime_error("Invalid live PCM16 chunk");
        auto samples = parse_pcm16(hex);
        SherpaOnnxOnlineStreamAcceptWaveform(active.get(), 16000, samples.data(), static_cast<int32_t>(samples.size()));
        while (SherpaOnnxIsOnlineStreamReady(online.get(), active.get()))
            SherpaOnnxDecodeOnlineStream(online.get(), active.get());
        return partial_result();
    }
    std::string finish_stream() {
        if (!active) throw std::runtime_error("No live stream to finish");
        const std::vector<float> silence(padding, 0.0f);
        SherpaOnnxOnlineStreamAcceptWaveform(active.get(), 16000, silence.data(), padding);
        SherpaOnnxOnlineStreamInputFinished(active.get());
        while (SherpaOnnxIsOnlineStreamReady(online.get(), active.get()))
            SherpaOnnxDecodeOnlineStream(online.get(), active.get());
        const auto result = partial_result();
        active.reset();
        return result;
    }
    std::string abort_stream() { active.reset(); return "{\"text\":\"\"}"; }

    std::string begin_clip() {
        if (collecting || active) throw std::runtime_error("Another speech stream is in progress");
        clip.clear();
        collecting = true;
        return "{\"text\":\"\"}";
    }
    std::string feed_clip(const std::string &hex) {
        if (!collecting) throw std::runtime_error("No clip input is active");
        auto chunk = parse_pcm16(hex);
        if (chunk.size() > 9600000 - clip.size())
            throw std::runtime_error("Clip exceeds ten minutes of PCM");
        clip.insert(clip.end(), chunk.begin(), chunk.end());
        return "{\"text\":\"\"}";
    }
    std::string finish_clip() {
        if (!collecting) throw std::runtime_error("No clip input is active");
        collecting = false;
        const auto result = decode_samples(clip.data(), static_cast<int32_t>(clip.size()));
        std::vector<float>().swap(clip);
        return result;
    }
    std::string abort_clip() {
        collecting = false;
        std::vector<float>().swap(clip);
        return "{\"text\":\"\"}";
    }

    std::string decode(const std::string &path) {
        Handle<SherpaOnnxWave, SherpaOnnxFreeWave> wave(SherpaOnnxReadWave(path.c_str()), SherpaOnnxFreeWave);
        if (!wave || wave->sample_rate != 16000)
            throw std::runtime_error("Expected mono 16 kHz PCM16 WAV");
        return decode_samples(wave->samples, wave->num_samples);
    }
    std::string decode_samples(const float *waveform, int32_t count) {
        if (!waveform || count <= 0 || count > 9600000)
            throw std::runtime_error("Expected between 1 and 9,600,000 PCM samples");
        if (offline) {
            // The author validates Sherpa 1.13.8 against NeMo at input gain 8.
            std::vector<float> samples(waveform, waveform + count);
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
        SherpaOnnxOnlineStreamAcceptWaveform(s.get(), 16000, waveform, count);
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
        Recognizer recognizer(args.at("--model"), fs::u8path(args.at("--weights")), threads);
        if (serve) {
            // One input path per line; one JSON result per line. EOF destroys weights.
            // The process is local, and stdout is reserved for this protocol.
            std::cout << "{\"ready\":true,\"pcm_protocol\":1}" << std::endl;
            std::string path;
            while (std::getline(std::cin, path)) {
                std::string json;
                if (path == "_HEX_CLIP_BEGIN_") {
                    json = recognizer.begin_clip();
                } else if (path == "_HEX_CLIP_END_") {
                    json = recognizer.finish_clip();
                } else if (path == "_HEX_CLIP_ABORT_") {
                    json = recognizer.abort_clip();
                } else if (path.rfind("_HEX_CLIP_PCM_", 0) == 0) {
                    json = recognizer.feed_clip(path.substr(14));
                } else if (path == "_HEX_BEGIN_") {
                    json = recognizer.begin_stream();
                } else if (path == "_HEX_END_") {
                    json = recognizer.finish_stream();
                } else if (path == "_HEX_ABORT_") {
                    json = recognizer.abort_stream();
                } else if (path.rfind("_HEX_PCM_", 0) == 0) {
                    json = recognizer.feed_stream(path.substr(9));
                } else {
                    if (path.size() > 32768) throw std::runtime_error("Invalid path");
                    json = recognizer.decode(path);
                }
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
