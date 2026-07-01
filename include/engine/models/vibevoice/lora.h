#pragma once

#include "engine/framework/assets/tensor_source.h"

#include <filesystem>
#include <memory>

namespace engine::models::vibevoice {

std::shared_ptr<const assets::TensorSource> make_lora_merged_tensor_source(
    std::shared_ptr<const assets::TensorSource> base,
    const std::filesystem::path & adapter_path,
    float scale_override);

}  // namespace engine::models::vibevoice
