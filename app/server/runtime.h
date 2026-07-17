#pragma once

#include "config.h"
#include "http.h"

#include "engine/framework/io/json.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/session.h"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace minitts::server {

class ServerState final : public IHttpHandler {
public:
    ServerState(ServerConfig config, std::filesystem::path request_base);

    HttpResponse handle(const HttpRequest & request) override;

private:
    struct LoadedModel {
        struct RuntimeVoicePreset {
            std::optional<std::string> voice_id;
            std::optional<engine::runtime::AudioBuffer> audio;
            std::optional<std::string> reference_text;
        };

        ServerModelConfig config;
        engine::runtime::TaskSpec task;
        std::unique_ptr<engine::runtime::ILoadedVoiceModel> model;
        std::unique_ptr<engine::runtime::IVoiceTaskSession> session;
        engine::runtime::IOfflineVoiceTaskSession * offline = nullptr;
        engine::runtime::IStreamingVoiceTaskSession * streaming = nullptr;
        std::unordered_map<std::string, RuntimeVoicePreset> voice_presets;
        std::optional<RuntimeVoicePreset> default_voice_preset;
        // timed_mutex so a request can bound how long it waits for the model.
        // busy_since_ms is the steady-clock time (ms) the current holder acquired the
        // lock, or 0 when idle; it lets a new request detect a run that has already
        // overrun and fail fast instead of queuing behind a wedged inference.
        std::timed_mutex mutex;
        std::atomic<std::int64_t> busy_since_ms{0};
    };

    // Holds model.mutex for the duration of a run and clears busy_since_ms when it
    // releases (including on exception), so the fail-fast check never sees a stale
    // timestamp. Move-only.
    class ModelRunLock {
    public:
        ModelRunLock() = default;
        ModelRunLock(LoadedModel & model, std::unique_lock<std::timed_mutex> lock)
            : model_(&model), lock_(std::move(lock)) {}
        ModelRunLock(ModelRunLock &&) = default;
        ModelRunLock & operator=(ModelRunLock &&) = default;
        ModelRunLock(const ModelRunLock &) = delete;
        ModelRunLock & operator=(const ModelRunLock &) = delete;
        ~ModelRunLock() {
            if (model_ != nullptr && lock_.owns_lock()) {
                model_->busy_since_ms.store(0, std::memory_order_release);
            }
        }

    private:
        LoadedModel * model_ = nullptr;
        std::unique_lock<std::timed_mutex> lock_;
    };

    // Acquire model.mutex for a run. Throws ServerBusyError (-> HTTP 503) when the
    // busy_timeout has elapsed, either because the current holder has already
    // overrun (fail fast, no wait) or because the wait itself timed out.
    ModelRunLock acquire_model_run(LoadedModel & model);

    void load_models();
    LoadedModel::RuntimeVoicePreset load_runtime_voice_preset(const ServerModelConfig::VoicePreset & preset) const;
    void load_voice_presets(LoadedModel & model) const;
    void ensure_model_loaded_locked(LoadedModel & model);
    LoadedModel & require_model(const engine::io::json::Value & body);
    const LoadedModel::RuntimeVoicePreset * select_voice_preset(
        const LoadedModel & model,
        const engine::io::json::Value & body,
        bool & voice_field_is_preset) const;
    engine::runtime::TaskRequest build_speech_request(
        const LoadedModel & model,
        const engine::io::json::Value & body) const;
    struct TimedTaskResult;
    TimedTaskResult run_model(LoadedModel & model, const engine::runtime::TaskRequest & request);
    TimedTaskResult run_streaming_model(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        const std::function<void(const engine::runtime::StreamEvent &)> & event_sink = {});
    HttpResponse handle_speech(const std::string & body_text);
    HttpResponse handle_speech_stream(
        LoadedModel & model,
        const engine::runtime::TaskRequest & request,
        const engine::io::json::Value & body);
    HttpResponse handle_transcription(const HttpRequest & request);
    HttpResponse handle_transcription_json(const std::string & body_text);
    HttpResponse handle_transcription_multipart(const std::string & body_text, const std::string & boundary);
    HttpResponse run_transcription(LoadedModel & model, const engine::runtime::TaskRequest & request);
    HttpResponse run_transcription_stream(LoadedModel & model, const engine::runtime::TaskRequest & request);
    HttpResponse handle_generic_run(const std::string & body_text);
    HttpResponse handle_generic_stream(const std::string & body_text);
    HttpResponse handle_voices(const HttpRequest & request) const;
    std::string models_json() const;

    ServerConfig config_;
    std::filesystem::path request_base_;
    std::vector<std::unique_ptr<LoadedModel>> models_;
    std::unordered_map<std::string, size_t> model_index_;
};

}  // namespace minitts::server
