#include "engine/models/vibevoice/lora.h"

#include "engine/framework/io/filesystem.h"
#include "engine/framework/io/json.h"

#include <cmath>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine::models::vibevoice {
namespace json = engine::io::json;
namespace {

constexpr std::string_view kLoraWeightsFile = "adapter_model.safetensors";
constexpr std::string_view kLoraConfigFile = "adapter_config.json";
constexpr std::string_view kPeftPrefix = "base_model.model.";
constexpr std::string_view kLanguageModelPrefix = "model.language_model.";
constexpr std::string_view kLoraASuffix = ".lora_A.weight";
constexpr std::string_view kLoraBSuffix = ".lora_B.weight";

struct LoraAdapterPaths {
    std::filesystem::path weights;
    std::optional<std::filesystem::path> config;
};

struct LoraTensorDelta {
    std::vector<float> a;
    std::vector<float> b;
    int64_t r = 0;
    int64_t in = 0;
    int64_t out = 0;
    float scale = 1.0F;
};

struct LoraTensorNames {
    std::optional<std::string> a_name;
    std::optional<std::string> b_name;
};

bool has_suffix(const std::string & name, std::string_view suffix) {
    return name.size() >= suffix.size() &&
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string strip_prefix(const std::string & name, std::string_view prefix) {
    if (name.rfind(prefix, 0) == 0) {
        return name.substr(prefix.size());
    }
    return name;
}

LoraAdapterPaths resolve_adapter_paths(const std::filesystem::path & adapter_path) {
    LoraAdapterPaths paths;
    if (engine::io::is_existing_directory(adapter_path)) {
        paths.weights = adapter_path / kLoraWeightsFile;
        const auto config = adapter_path / kLoraConfigFile;
        if (engine::io::is_existing_file(config)) {
            paths.config = config;
        }
    } else if (engine::io::is_existing_file(adapter_path)) {
        paths.weights = adapter_path;
        const auto config = adapter_path.parent_path() / kLoraConfigFile;
        if (engine::io::is_existing_file(config)) {
            paths.config = config;
        }
    } else {
        throw std::runtime_error("VibeVoice LoRA path does not exist: " + adapter_path.string());
    }
    if (!engine::io::is_existing_file(paths.weights)) {
        throw std::runtime_error("VibeVoice LoRA adapter weights not found: " + paths.weights.string());
    }
    return paths;
}

float resolve_adapter_scale(const std::optional<std::filesystem::path> & config_path, float scale_override) {
    if (scale_override > 0.0F) {
        return scale_override;
    }
    if (!config_path.has_value()) {
        throw std::runtime_error(
            "VibeVoice LoRA has no adapter_config.json; pass vibevoice.lora_scale to set the merge scale");
    }
    const auto root = json::parse_file(*config_path);
    const auto rank = json::require_i64(root, "r");
    if (rank <= 0) {
        throw std::runtime_error("VibeVoice LoRA adapter_config.json has non-positive r");
    }
    const float alpha = json::optional_f32(root, "lora_alpha", static_cast<float>(rank));
    const bool use_rslora = json::optional_bool(root, "use_rslora", false);
    const float denom = use_rslora ? std::sqrt(static_cast<float>(rank)) : static_cast<float>(rank);
    return alpha / denom;
}

std::string resolve_base_weight_name(const assets::TensorSource & base, const std::string & module_path) {
    const std::string prefixed = std::string(kLanguageModelPrefix) + module_path + ".weight";
    if (base.has_tensor(prefixed)) {
        return prefixed;
    }
    const std::string bare = module_path + ".weight";
    if (base.has_tensor(bare)) {
        return bare;
    }
    throw std::runtime_error("VibeVoice LoRA targets a module with no matching base weight: " + module_path);
}

std::vector<float> merged_f32_values(
    const assets::TensorSource & base,
    const std::string & name,
    const LoraTensorDelta & delta) {
    auto values = base.require_f32(name, std::optional<std::vector<int64_t>>({delta.out, delta.in}));
    // values[o, i] += scale * sum_k B[o, k] * A[k, i]
    for (int64_t o = 0; o < delta.out; ++o) {
        const int64_t row = o * delta.in;
        for (int64_t k = 0; k < delta.r; ++k) {
            const float b = delta.scale * delta.b[static_cast<size_t>(o * delta.r + k)];
            if (b == 0.0F) {
                continue;
            }
            const float * a_row = delta.a.data() + static_cast<size_t>(k * delta.in);
            for (int64_t i = 0; i < delta.in; ++i) {
                values[static_cast<size_t>(row + i)] += b * a_row[i];
            }
        }
    }
    return values;
}

class LoraMergedTensorSource final : public assets::TensorSource {
public:
    LoraMergedTensorSource(
        std::shared_ptr<const assets::TensorSource> base,
        std::unordered_map<std::string, LoraTensorDelta> deltas)
        : base_(std::move(base)),
          deltas_(std::move(deltas)) {}

    const std::filesystem::path & source_path() const noexcept override {
        return base_->source_path();
    }

    bool has_tensor(std::string_view name) const noexcept override {
        return base_->has_tensor(name);
    }

    assets::TensorMetadata require_metadata(std::string_view name) const override {
        return base_->require_metadata(name);
    }

    std::vector<assets::TensorMetadata> tensors() const override {
        return base_->tensors();
    }

    void release_storage() const override {
        base_->release_storage();
    }

    assets::RawTensorData require_tensor_data(std::string_view name) const override {
        const auto delta = deltas_.find(std::string(name));
        if (delta == deltas_.end()) {
            return base_->require_tensor_data(name);
        }
        const std::string key(name);
        const auto values = merged_f32_values(*base_, key, delta->second);
        assets::RawTensorData raw;
        raw.metadata = assets::TensorMetadata{key, "F32", {delta->second.out, delta->second.in}};
        raw.bytes.resize(values.size() * sizeof(float));
        std::memcpy(raw.bytes.data(), values.data(), raw.bytes.size());
        return raw;
    }

    std::vector<float> require_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        const auto delta = deltas_.find(std::string(name));
        if (delta == deltas_.end()) {
            return base_->require_f32(name, expected_shape);
        }
        return merged_f32_values(*base_, std::string(name), delta->second);
    }

    std::optional<std::vector<float>> optional_f32(
        std::string_view name,
        const std::optional<std::vector<int64_t>> & expected_shape) const override {
        if (!has_tensor(name)) {
            return std::nullopt;
        }
        return require_f32(name, expected_shape);
    }

    int64_t require_i64_scalar(std::string_view name) const override {
        return base_->require_i64_scalar(name);
    }

private:
    std::shared_ptr<const assets::TensorSource> base_;
    std::unordered_map<std::string, LoraTensorDelta> deltas_;
};

std::unordered_map<std::string, LoraTensorNames> collect_lora_tensor_names(const assets::TensorSource & adapter) {
    std::unordered_map<std::string, LoraTensorNames> names;
    for (const auto & metadata : adapter.tensors()) {
        const std::string stripped = strip_prefix(metadata.name, kPeftPrefix);
        if (has_suffix(stripped, kLoraASuffix)) {
            names[stripped.substr(0, stripped.size() - kLoraASuffix.size())].a_name = metadata.name;
        } else if (has_suffix(stripped, kLoraBSuffix)) {
            names[stripped.substr(0, stripped.size() - kLoraBSuffix.size())].b_name = metadata.name;
        }
    }
    return names;
}

LoraTensorDelta load_lora_delta(
    const assets::TensorSource & base,
    const assets::TensorSource & adapter,
    const std::string & module_path,
    const LoraTensorNames & names,
    const std::string & base_name,
    float scale) {
    if (!names.a_name.has_value() || !names.b_name.has_value()) {
        throw std::runtime_error("VibeVoice LoRA module is missing an A/B pair: " + module_path);
    }
    const auto a_meta = adapter.require_metadata(*names.a_name);
    const auto b_meta = adapter.require_metadata(*names.b_name);
    if (a_meta.shape.size() != 2 || b_meta.shape.size() != 2) {
        throw std::runtime_error("VibeVoice LoRA A/B tensors must be rank-2: " + module_path);
    }
    LoraTensorDelta delta;
    delta.r = a_meta.shape[0];
    delta.in = a_meta.shape[1];
    delta.out = b_meta.shape[0];
    delta.scale = scale;
    if (b_meta.shape[1] != delta.r) {
        throw std::runtime_error("VibeVoice LoRA A/B rank mismatch for " + module_path);
    }
    if (base.require_metadata(base_name).shape != std::vector<int64_t>({delta.out, delta.in})) {
        throw std::runtime_error(
            "VibeVoice LoRA shape mismatch for " + base_name + " (adapter is trained for a different model size)");
    }
    delta.a = adapter.require_f32(*names.a_name, std::optional<std::vector<int64_t>>({delta.r, delta.in}));
    delta.b = adapter.require_f32(*names.b_name, std::optional<std::vector<int64_t>>({delta.out, delta.r}));
    return delta;
}

}  // namespace

std::shared_ptr<const assets::TensorSource> make_lora_merged_tensor_source(
    std::shared_ptr<const assets::TensorSource> base,
    const std::filesystem::path & adapter_path,
    float scale_override) {
    if (base == nullptr) {
        throw std::runtime_error("VibeVoice LoRA merge requires a base tensor source");
    }
    const auto paths = resolve_adapter_paths(adapter_path);
    const float scale = resolve_adapter_scale(paths.config, scale_override);
    const auto adapter = assets::open_tensor_source(paths.weights);

    const auto tensor_names = collect_lora_tensor_names(*adapter);
    if (tensor_names.empty()) {
        throw std::runtime_error("VibeVoice LoRA adapter contains no lora_A/lora_B tensors: " + paths.weights.string());
    }

    std::unordered_map<std::string, LoraTensorDelta> deltas;
    deltas.reserve(tensor_names.size());
    for (const auto & [module_path, names] : tensor_names) {
        const std::string base_name = resolve_base_weight_name(*base, module_path);
        deltas.emplace(base_name, load_lora_delta(*base, *adapter, module_path, names, base_name, scale));
    }

    return std::make_shared<LoraMergedTensorSource>(std::move(base), std::move(deltas));
}

}  // namespace engine::models::vibevoice
