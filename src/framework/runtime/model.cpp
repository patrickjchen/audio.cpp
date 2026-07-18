#include "engine/framework/runtime/model.h"

#include "engine/framework/assets/model_package.h"
#include "engine/framework/io/filesystem.h"

#include <unordered_map>
#include <stdexcept>

namespace engine::runtime {

namespace {

std::string normalized_candidate_key(const std::string & relative_candidate) {
    return std::filesystem::path(relative_candidate).generic_string();
}

std::string candidate_stem_or_filename(const std::string & relative_candidate) {
    const auto path = std::filesystem::path(relative_candidate);
    if (!path.stem().empty()) {
        return path.stem().string();
    }
    return path.filename().string();
}

}

std::vector<NamedAsset> discover_named_assets(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates) {
    std::unordered_map<std::string, int> stem_counts;
    for (const auto & candidate : relative_candidates) {
        ++stem_counts[candidate_stem_or_filename(candidate)];
    }

    std::vector<NamedAsset> assets;
    assets.reserve(relative_candidates.size());
    for (const auto & candidate : relative_candidates) {
        const auto path = root / candidate;
        if (engine::io::is_existing_file(path)) {
            const auto base_id = candidate_stem_or_filename(candidate);
            const auto id = stem_counts[base_id] == 1 ? base_id : normalized_candidate_key(candidate);
            assets.push_back({id, std::filesystem::weakly_canonical(path)});
        }
    }
    return assets;
}

std::vector<NamedAsset> discover_named_assets_from_package_spec(
    const std::filesystem::path & model_path,
    const std::filesystem::path & spec_path,
    engine::assets::ModelPackageResourceKind kind) {
    const auto resources = engine::assets::discover_resources_from_package_spec(model_path, spec_path, kind);
    std::vector<NamedAsset> assets;
    assets.reserve(resources.size());
    for (const auto & resource : resources) {
        assets.push_back({resource.id, resource.path});
    }
    return assets;
}

const NamedAsset * find_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::string & id) noexcept {
    for (const auto & asset : assets) {
        if (asset.id == id) {
            return &asset;
        }
    }
    return nullptr;
}

const NamedAsset & require_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::string & id,
    const std::string & role) {
    const auto * asset = find_named_asset(assets, id);
    if (asset == nullptr) {
        throw std::runtime_error("unknown " + role + " id: " + id);
    }
    return *asset;
}

const NamedAsset * select_named_asset(
    const std::vector<NamedAsset> & assets,
    const std::optional<std::string> & id,
    const std::string & role) {
    if (id.has_value()) {
        return &require_named_asset(assets, *id, role);
    }
    if (assets.empty()) {
        return nullptr;
    }
    if (assets.size() > 1) {
        throw std::runtime_error("multiple " + role + " candidates found; specify --" + role);
    }
    return &assets.front();
}

namespace {

bool contains_token(const std::string & family, const char * token) {
    return family.find(token) != std::string::npos;
}

bool ends_with(const std::string & family, const char * suffix) {
    const std::string needle(suffix);
    return family.size() >= needle.size()
        && family.compare(family.size() - needle.size(), needle.size(), needle) == 0;
}

CapabilitySet caps_for_task(VoiceTaskKind task, bool streaming = false) {
    CapabilitySet caps;
    TaskCapability row;
    row.task = task;
    row.modes.push_back(RunMode::Offline);
    if (streaming) {
        row.modes.push_back(RunMode::Streaming);
    }
    caps.supported_tasks.push_back(std::move(row));
    return caps;
}

}  // namespace

CapabilitySet default_advertised_capabilities_for_family(const std::string & family) {
    const std::string key = family;
    if (contains_token(key, "forced_aligner") || ends_with(key, "_aligner") || ends_with(key, "_align")) {
        return caps_for_task(VoiceTaskKind::Alignment);
    }
    if (ends_with(key, "_asr") || ends_with(key, "_stt") || key == "parakeet_tdt" || key == "whisper") {
        const bool streaming = contains_token(key, "higgs") || contains_token(key, "nemotron")
            || contains_token(key, "vibevoice");
        return caps_for_task(VoiceTaskKind::Asr, streaming);
    }
    if (contains_token(key, "vad")) {
        return caps_for_task(VoiceTaskKind::Vad, contains_token(key, "silero"));
    }
    if (contains_token(key, "diar") || contains_token(key, "sortformer")) {
        return caps_for_task(VoiceTaskKind::Diarization);
    }
    if (contains_token(key, "demucs") || contains_token(key, "roformer")) {
        return caps_for_task(VoiceTaskKind::SourceSeparation);
    }
    if (key == "stable_audio" || key == "ace_step" || key == "heartmula" || ends_with(key, "_gen")) {
        return caps_for_task(VoiceTaskKind::AudioGeneration);
    }
    if (key == "seed_vc" || key == "vevo2" || ends_with(key, "_vc")) {
        CapabilitySet caps = caps_for_task(VoiceTaskKind::VoiceConversion);
        if (key == "vevo2") {
            caps.supported_tasks.push_back(TaskCapability{VoiceTaskKind::Tts, {RunMode::Offline}});
            caps.supported_tasks.push_back(TaskCapability{VoiceTaskKind::Svc, {RunMode::Offline}});
        }
        return caps;
    }
    if (key == "miocodec" || ends_with(key, "_codec")) {
        // Codec helpers are exposed via tasks/run style surfaces.
        return caps_for_task(VoiceTaskKind::VoiceConversion);
    }
    if (contains_token(key, "chatterbox")) {
        CapabilitySet caps = caps_for_task(VoiceTaskKind::Tts);
        caps.supported_tasks.push_back(TaskCapability{VoiceTaskKind::VoiceConversion, {RunMode::Offline}});
        return caps;
    }
    if (contains_token(key, "vibevoice") && contains_token(key, "asr")) {
        return caps_for_task(VoiceTaskKind::Asr, true);
    }
    if (key == "qwen3_tts" || (contains_token(key, "tts") && contains_token(key, "voice_design"))) {
        CapabilitySet caps = caps_for_task(VoiceTaskKind::Tts);
        caps.supported_tasks.push_back(TaskCapability{VoiceTaskKind::VoiceDesign, {RunMode::Offline}});
        return caps;
    }
    // Default speech families (tts / omnivoice / vibevoice / moss / etc.)
    if (contains_token(key, "tts") || contains_token(key, "kokoro") || contains_token(key, "voxcpm")
        || contains_token(key, "omnivoice") || contains_token(key, "supertonic")
        || contains_token(key, "miotts") || contains_token(key, "pocket")
        || contains_token(key, "irodori") || contains_token(key, "moss")
        || contains_token(key, "index_tts") || contains_token(key, "vibevoice")) {
        const bool streaming = contains_token(key, "voxcpm") || contains_token(key, "supertonic");
        return caps_for_task(VoiceTaskKind::Tts, streaming);
    }
    return {};
}

std::string default_instructions_policy_for_family(const std::string & family) {
    if (contains_token(family, "irodori")) {
        return "caption_option";
    }
    if (contains_token(family, "voxcpm")) {
        return "text_prefix";
    }
    if (family == "omnivoice" || contains_token(family, "omnivoice")) {
        return "soft_tags";
    }
    if (contains_token(family, "qwen3_tts") || contains_token(family, "voice_design")) {
        return "openai_instruct";
    }
    const auto caps = default_advertised_capabilities_for_family(family);
    for (const auto & task : caps.supported_tasks) {
        if (task.task == VoiceTaskKind::Tts || task.task == VoiceTaskKind::VoiceDesign
            || task.task == VoiceTaskKind::VoiceCloning) {
            return "openai_instruct";
        }
    }
    return "none";
}

std::vector<std::string> default_api_endpoints_for_capabilities(const CapabilitySet & capabilities) {
    bool has_asr = false;
    bool has_speech = false;
    bool has_other = false;
    for (const auto & task : capabilities.supported_tasks) {
        switch (task.task) {
        case VoiceTaskKind::Asr:
            has_asr = true;
            break;
        case VoiceTaskKind::Tts:
        case VoiceTaskKind::VoiceCloning:
        case VoiceTaskKind::VoiceDesign:
            has_speech = true;
            break;
        default:
            has_other = true;
            break;
        }
    }
    if (has_asr && !has_speech && !has_other) {
        return {"/v1/audio/transcriptions"};
    }
    if (has_speech && !has_other) {
        return {"/v1/audio/speech"};
    }
    if (has_speech && has_other) {
        return {"/v1/tasks/run", "/v1/audio/speech"};
    }
    return {"/v1/tasks/run"};
}

CapabilitySet IVoiceModelLoader::advertised_capabilities() const {
    return default_advertised_capabilities_for_family(family());
}

std::string IVoiceModelLoader::advertised_instructions_policy() const {
    return default_instructions_policy_for_family(family());
}

std::vector<std::string> IVoiceModelLoader::advertised_api_endpoints() const {
    return default_api_endpoints_for_capabilities(advertised_capabilities());
}

LoaderAdvertisement IVoiceModelLoader::advertise() const {
    LoaderAdvertisement row;
    row.family = family();
    row.capabilities = advertised_capabilities();
    row.instructions_policy = advertised_instructions_policy();
    row.api_endpoints = advertised_api_endpoints();
    return row;
}

}  // namespace engine::runtime
