/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"

namespace cutlass {
namespace gemm {

enum SoftwareCacheLineState : int {
  kSoftwareCacheLineInvalid = 0,
  kSoftwareCacheLineFilling = 1,
  kSoftwareCacheLineValid = 2
};

struct SoftwareCacheDescriptor {
  void const* remote_base = nullptr;
  int64_t remote_bytes = 0;
  void* local_base = nullptr;
  int64_t local_minus_remote_offset = 0;
  int* tile_states = nullptr;
  int tile_shape_contiguous = 0;
  int tile_shape_strided = 0;
  int tile_count_contiguous = 0;
  int tile_count_strided = 0;
  int enabled = 0;
  // log2 of tile_shape_* for fast right-shift division (only valid when shape is power-of-2)
  int tile_shape_contiguous_shift = 0;
  int tile_shape_strided_shift = 0;
  // When set, all tiles are pre-loaded and Valid. Skips every per-tile state check,
  // including tile_states reads and cache_operand_stage syncthreads.
  int all_tiles_valid = 0;

  CUTLASS_HOST_DEVICE
  bool is_enabled() const {
    return enabled &&
      remote_base &&
      remote_bytes > 0 &&
      local_base &&
      tile_states &&
      tile_shape_contiguous > 0 &&
      tile_shape_strided > 0 &&
      tile_count_contiguous > 0 &&
      tile_count_strided > 0;
  }
};

}  // namespace gemm
}  // namespace cutlass
