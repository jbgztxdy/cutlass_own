#pragma once

#include <cuda/atomic>

#include "cutlass/arch/memory.h"
#include "cutlass/coord.h"
#include "cutlass/gemm/threadblock/default_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

namespace cutlass {
namespace layout {

struct RowMajorRingBuffer {
  static int const kRank = 2;
  static int const kStrideRank = 1;

  using Index = int32_t;
  using LongIndex = int64_t;
  using TensorCoord = MatrixCoord;
  using Stride = Coord<kStrideRank, LongIndex>;

  Stride stride_;
  int tile_k = 0;
  int slot_count = 0;
  LongIndex slot_stride = 0;
  int *ready_flags = nullptr;

  CUTLASS_HOST_DEVICE
  RowMajorRingBuffer() = default;

  CUTLASS_HOST_DEVICE
  RowMajorRingBuffer(
      LongIndex ldm,
      int slot_count_,
      LongIndex slot_stride_,
      int *ready_flags_ = nullptr)
      : stride_(ldm),
        tile_k(int(ldm)),
        slot_count(slot_count_),
        slot_stride(slot_stride_),
        ready_flags(ready_flags_) {}

  CUTLASS_HOST_DEVICE
  static RowMajorRingBuffer packed(MatrixCoord const &extent) {
    return RowMajorRingBuffer(extent.column(), 1, LongIndex(extent.product()));
  }

  CUTLASS_HOST_DEVICE
  Stride const &stride() const {
    return stride_;
  }

  CUTLASS_HOST_DEVICE
  Stride &stride() {
    return stride_;
  }

  CUTLASS_HOST_DEVICE
  LongIndex stride(int rank) const {
    return stride_.at(rank);
  }

  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    int k_tile = coord.column() / tile_k;
    int inner_k = coord.column() % tile_k;
    return LongIndex(k_tile % slot_count) * slot_stride +
           LongIndex(coord.row()) * stride_[0] + inner_k;
  }
};

struct ColumnMajorRingBuffer {
  static int const kRank = 2;
  static int const kStrideRank = 1;

  using Index = int32_t;
  using LongIndex = int64_t;
  using TensorCoord = MatrixCoord;
  using Stride = Coord<kStrideRank, LongIndex>;

  Stride stride_;
  int tile_k = 0;
  int slot_count = 0;
  LongIndex slot_stride = 0;
  int *ready_flags = nullptr;

  CUTLASS_HOST_DEVICE
  ColumnMajorRingBuffer() = default;

  CUTLASS_HOST_DEVICE
  ColumnMajorRingBuffer(
      LongIndex ldm,
      int slot_count_,
      LongIndex slot_stride_,
      int *ready_flags_ = nullptr)
      : stride_(ldm),
        tile_k(int(ldm)),
        slot_count(slot_count_),
        slot_stride(slot_stride_),
        ready_flags(ready_flags_) {}

  CUTLASS_HOST_DEVICE
  static ColumnMajorRingBuffer packed(MatrixCoord const &extent) {
    return ColumnMajorRingBuffer(extent.row(), 1, LongIndex(extent.product()));
  }

  CUTLASS_HOST_DEVICE
  Stride const &stride() const {
    return stride_;
  }

  CUTLASS_HOST_DEVICE
  Stride &stride() {
    return stride_;
  }

  CUTLASS_HOST_DEVICE
  LongIndex stride(int rank) const {
    return stride_.at(rank);
  }

  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    int k_tile = coord.row() / tile_k;
    int inner_k = coord.row() % tile_k;
    return LongIndex(k_tile % slot_count) * slot_stride +
           LongIndex(coord.column()) * stride_[0] + inner_k;
  }
};

// A operand: logical shape (M, K), column-major, K advances along columns.
struct ColumnMajorRingBufferA {
  static int const kRank = 2;
  static int const kStrideRank = 1;

  using Index = int32_t;
  using LongIndex = int64_t;
  using TensorCoord = MatrixCoord;
  using Stride = Coord<kStrideRank, LongIndex>;

  Stride stride_;
  int tile_k = 0;
  int slot_count = 0;
  LongIndex slot_stride = 0;
  int *ready_flags = nullptr;

  CUTLASS_HOST_DEVICE
  ColumnMajorRingBufferA() = default;

  CUTLASS_HOST_DEVICE
  ColumnMajorRingBufferA(
      LongIndex physical_stride,
      int tile_k_,
      int slot_count_,
      LongIndex slot_stride_,
      int *ready_flags_ = nullptr)
      : stride_(physical_stride),
        tile_k(tile_k_),
        slot_count(slot_count_),
        slot_stride(slot_stride_),
        ready_flags(ready_flags_) {}

  CUTLASS_HOST_DEVICE
  Stride const &stride() const {
    return stride_;
  }

  CUTLASS_HOST_DEVICE
  Stride &stride() {
    return stride_;
  }

  CUTLASS_HOST_DEVICE
  LongIndex stride(int rank) const {
    return stride_.at(rank);
  }

  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    int k_tile = coord.column() / tile_k;
    int inner_k = coord.column() % tile_k;
    return LongIndex(k_tile % slot_count) * slot_stride +
           LongIndex(inner_k) * stride_[0] + coord.row();
  }
};

// B operand: logical shape (K, N), row-major, K advances along rows.
struct RowMajorRingBufferB {
  static int const kRank = 2;
  static int const kStrideRank = 1;

  using Index = int32_t;
  using LongIndex = int64_t;
  using TensorCoord = MatrixCoord;
  using Stride = Coord<kStrideRank, LongIndex>;

  Stride stride_;
  int tile_k = 0;
  int slot_count = 0;
  LongIndex slot_stride = 0;
  int *ready_flags = nullptr;

  CUTLASS_HOST_DEVICE
  RowMajorRingBufferB() = default;

  CUTLASS_HOST_DEVICE
  RowMajorRingBufferB(
      LongIndex physical_stride,
      int tile_k_,
      int slot_count_,
      LongIndex slot_stride_,
      int *ready_flags_ = nullptr)
      : stride_(physical_stride),
        tile_k(tile_k_),
        slot_count(slot_count_),
        slot_stride(slot_stride_),
        ready_flags(ready_flags_) {}

  CUTLASS_HOST_DEVICE
  Stride const &stride() const {
    return stride_;
  }

  CUTLASS_HOST_DEVICE
  Stride &stride() {
    return stride_;
  }

  CUTLASS_HOST_DEVICE
  LongIndex stride(int rank) const {
    return stride_.at(rank);
  }

  CUTLASS_HOST_DEVICE
  LongIndex operator()(TensorCoord const &coord) const {
    int k_tile = coord.row() / tile_k;
    int inner_k = coord.row() % tile_k;
    return LongIndex(k_tile % slot_count) * slot_stride +
           LongIndex(inner_k) * stride_[0] + coord.column();
  }
};

}  // namespace layout
}  // namespace cutlass
namespace cutlass {
namespace transform {
namespace threadblock {

namespace detail {

template <typename Shape_, typename Element_, typename ThreadMap_, typename AccessType_>
class RingPanelTileAccessIteratorPitchLinear {
 public:
  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = typename Layout::TensorCoord;
  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingPredicates = PredicatedTileAccessIteratorPredicates<
      Shape, Element, Layout, 0, ThreadMap, AccessType>;

  using Mask = typename UnderlyingPredicates::Mask;

  static int const kAccessesPerVector =
      ThreadMap::kElementsPerAccess / AccessType::kElements;

  struct Params {
    LongIndex physical_stride = 0;
    int tile_k = 0;
    int slot_count = 0;
    LongIndex slot_stride = 0;
    int *ready_flags = nullptr;

    CUTLASS_HOST_DEVICE
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(
        LongIndex physical_stride_,
        int tile_k_,
        int slot_count_,
        LongIndex slot_stride_,
        int *ready_flags_)
        : physical_stride(physical_stride_),
          tile_k(tile_k_),
          slot_count(slot_count_),
          slot_stride(slot_stride_),
          ready_flags(ready_flags_) {}
  };

 private:
  using BytePointer = char *;
  using Atomic = cuda::atomic<int, cuda::thread_scope_system>;

  UnderlyingPredicates the_predicates;
  Params params_;
  BytePointer pointer_ = nullptr;
  TensorCoord coord_offset_;
  bool is_residue_tile_ = true;
  int thread_idx_ = 0;
  int waited_k_tile_ = -1;
  int max_k_tiles_ = 0;

 public:
  CUTLASS_HOST_DEVICE
  RingPanelTileAccessIteratorPitchLinear() = default;

  CUTLASS_HOST_DEVICE
  RingPanelTileAccessIteratorPitchLinear(
      Params const &params,
      Pointer pointer,
      TensorCoord extent,
      int thread_id,
      TensorCoord const &threadblock_offset)
      : the_predicates(extent),
        params_(params),
        pointer_(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),
        coord_offset_(threadblock_offset),
        thread_idx_(thread_id),
        max_k_tiles_(params.tile_k ? (extent.contiguous() + params.tile_k - 1) / params.tile_k
                                   : 0) {
    the_predicates.set_predicates(thread_id, threadblock_offset);
    coord_offset_ = the_predicates.thread_offset_;
    wait_for_k_tile(params.tile_k ? threadblock_offset.contiguous() / params.tile_k
                                  : 0);
  }

  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    the_predicates.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex) {}

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    if (is_residue_tile_) {
      the_predicates.thread_offset_ += the_predicates.residue_offset_;
      the_predicates.compute_predicates_(the_predicates.extent_, true);

      coord_offset_.contiguous() =
          the_predicates.thread_offset_.contiguous() +
          Shape::kContiguous * (tile_offset.contiguous() - 1);
      coord_offset_.strided() = the_predicates.thread_offset_.strided() +
                                Shape::kStrided * tile_offset.strided();
    } else {
      coord_offset_.contiguous() +=
          Shape::kContiguous * tile_offset.contiguous();
      coord_offset_.strided() += Shape::kStrided * tile_offset.strided();
    }

    is_residue_tile_ = false;

    if (tile_offset.contiguous() > 0) {
      int next_k_tile = coord_offset_.contiguous() / params_.tile_k;
      wait_for_k_tile(next_k_tile);
    }
  }

  CUTLASS_DEVICE
  AccessType *get() const {
    if (!the_predicates.valid()) {
      return nullptr;
    }

    Index coord_contig = coord_offset_.contiguous() +
                         the_predicates.iteration_contiguous_ *
                             ThreadMap::Delta::kContiguous +
                         the_predicates.iteration_vector_ * AccessType::kElements;
    Index coord_strided = coord_offset_.strided() +
                          the_predicates.iteration_strided_ *
                              ThreadMap::Delta::kStrided;

    int k_tile = coord_contig / params_.tile_k;
    int inner_k = coord_contig % params_.tile_k;

    LongIndex offset = LongIndex(k_tile % params_.slot_count) * params_.slot_stride +
                       LongIndex(coord_strided) * params_.physical_stride +
                       inner_k;

    return reinterpret_cast<AccessType *>(pointer_ + OffsetBytes<Element>(offset));
  }

  CUTLASS_HOST_DEVICE
  RingPanelTileAccessIteratorPitchLinear &operator++() {
    ++the_predicates.iteration_vector_;
    if (the_predicates.iteration_vector_ < kAccessesPerVector) {
      return *this;
    }

    the_predicates.iteration_vector_ = 0;
    ++the_predicates.iteration_contiguous_;
    if (the_predicates.iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
      return *this;
    }

    the_predicates.iteration_contiguous_ = 0;
    ++the_predicates.iteration_strided_;
    if (the_predicates.iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }

    the_predicates.iteration_strided_ = 0;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  RingPanelTileAccessIteratorPitchLinear operator++(int) {
    RingPanelTileAccessIteratorPitchLinear self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    the_predicates.clear_mask(enable);
  }

  CUTLASS_HOST_DEVICE
  void enable_mask() {
    the_predicates.enable_mask();
  }

  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    the_predicates.set_mask(mask);
  }

  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    the_predicates.get_mask(mask);
  }

  CUTLASS_HOST_DEVICE
  bool valid() const {
    return the_predicates.valid();
  }

  CUTLASS_DEVICE
  void wait_for_current_k_tile() {
    if (!params_.tile_k) {
      return;
    }
    wait_for_k_tile(coord_offset_.contiguous() / params_.tile_k);
  }

 private:
  CUTLASS_DEVICE
  void wait_for_k_tile(int k_tile) {
    if (!params_.ready_flags || k_tile <= waited_k_tile_ || k_tile >= max_k_tiles_) {
      return;
    }

    if ((thread_idx_ & 31) == 0) {
      int slot = k_tile % params_.slot_count;
      int target = k_tile + 1;
      Atomic *ready_ptr = reinterpret_cast<Atomic *>(params_.ready_flags + slot);
      int observed = ready_ptr->load(cuda::std::memory_order_acquire);
      while (observed < target) {
        ready_ptr->wait(observed, cuda::std::memory_order_acquire);
        observed = ready_ptr->load(cuda::std::memory_order_acquire);
      }
    }
    __syncwarp();
    waited_k_tile_ = k_tile;
  }
};

template <typename Shape_, typename Element_, typename ThreadMap_, typename AccessType_>
class RingPanelTileAccessIteratorPitchLinearKStrided {
 public:
  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = typename Layout::TensorCoord;
  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingPredicates = PredicatedTileAccessIteratorPredicates<
      Shape, Element, Layout, 0, ThreadMap, AccessType>;

  using Mask = typename UnderlyingPredicates::Mask;

  static int const kAccessesPerVector =
      ThreadMap::kElementsPerAccess / AccessType::kElements;

  struct Params {
    LongIndex physical_stride = 0;
    int tile_k = 0;
    int slot_count = 0;
    LongIndex slot_stride = 0;
    int *ready_flags = nullptr;

    CUTLASS_HOST_DEVICE
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(
        LongIndex physical_stride_,
        int tile_k_,
        int slot_count_,
        LongIndex slot_stride_,
        int *ready_flags_)
        : physical_stride(physical_stride_),
          tile_k(tile_k_),
          slot_count(slot_count_),
          slot_stride(slot_stride_),
          ready_flags(ready_flags_) {}
  };

 private:
  using BytePointer = char *;
  using Atomic = cuda::atomic<int, cuda::thread_scope_system>;

  UnderlyingPredicates the_predicates;
  Params params_;
  BytePointer pointer_ = nullptr;
  TensorCoord coord_offset_;
  bool is_residue_tile_ = true;
  int thread_idx_ = 0;
  int waited_k_tile_ = -1;
  int max_k_tiles_ = 0;

 public:
  CUTLASS_HOST_DEVICE
  RingPanelTileAccessIteratorPitchLinearKStrided() = default;

  CUTLASS_HOST_DEVICE
  RingPanelTileAccessIteratorPitchLinearKStrided(
      Params const &params,
      Pointer pointer,
      TensorCoord extent,
      int thread_id,
      TensorCoord const &threadblock_offset)
      : the_predicates(extent),
        params_(params),
        pointer_(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),
        coord_offset_(threadblock_offset),
        thread_idx_(thread_id),
        max_k_tiles_(params.tile_k ? (extent.contiguous() + params.tile_k - 1) / params.tile_k
                                   : 0) {
    the_predicates.set_predicates(thread_id, threadblock_offset);
    coord_offset_ = the_predicates.thread_offset_;
    wait_for_k_tile(params.tile_k ? threadblock_offset.contiguous() / params.tile_k
                                  : 0);
  }

  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    the_predicates.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex) {}

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    if (is_residue_tile_) {
      the_predicates.thread_offset_ += the_predicates.residue_offset_;
      the_predicates.compute_predicates_(the_predicates.extent_, true);

      coord_offset_.contiguous() =
          the_predicates.thread_offset_.contiguous() +
          Shape::kContiguous * (tile_offset.contiguous() - 1);
      coord_offset_.strided() = the_predicates.thread_offset_.strided() +
                                Shape::kStrided * tile_offset.strided();
    } else {
      coord_offset_.contiguous() +=
          Shape::kContiguous * tile_offset.contiguous();
      coord_offset_.strided() += Shape::kStrided * tile_offset.strided();
    }

    is_residue_tile_ = false;

    if (tile_offset.contiguous() > 0) {
      int next_k_tile = coord_offset_.contiguous() / params_.tile_k;
      wait_for_k_tile(next_k_tile);
    }
  }

  CUTLASS_DEVICE
  AccessType *get() const {
    if (!the_predicates.valid()) {
      return nullptr;
    }

    Index coord_contig = coord_offset_.contiguous() +
                         the_predicates.iteration_contiguous_ *
                             ThreadMap::Delta::kContiguous +
                         the_predicates.iteration_vector_ * AccessType::kElements;
    Index coord_strided = coord_offset_.strided() +
                          the_predicates.iteration_strided_ *
                              ThreadMap::Delta::kStrided;

    int k_tile = coord_contig / params_.tile_k;
    int inner_k = coord_contig % params_.tile_k;

    LongIndex offset = LongIndex(k_tile % params_.slot_count) * params_.slot_stride +
                       LongIndex(inner_k) * params_.physical_stride +
                       coord_strided;

    return reinterpret_cast<AccessType *>(pointer_ + OffsetBytes<Element>(offset));
  }

  CUTLASS_HOST_DEVICE
  RingPanelTileAccessIteratorPitchLinearKStrided &operator++() {
    ++the_predicates.iteration_vector_;
    if (the_predicates.iteration_vector_ < kAccessesPerVector) {
      return *this;
    }

    the_predicates.iteration_vector_ = 0;
    ++the_predicates.iteration_contiguous_;
    if (the_predicates.iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
      return *this;
    }

    the_predicates.iteration_contiguous_ = 0;
    ++the_predicates.iteration_strided_;
    if (the_predicates.iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }

    the_predicates.iteration_strided_ = 0;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  RingPanelTileAccessIteratorPitchLinearKStrided operator++(int) {
    RingPanelTileAccessIteratorPitchLinearKStrided self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    the_predicates.clear_mask(enable);
  }

  CUTLASS_HOST_DEVICE
  void enable_mask() {
    the_predicates.enable_mask();
  }

  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    the_predicates.set_mask(mask);
  }

  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    the_predicates.get_mask(mask);
  }

  CUTLASS_HOST_DEVICE
  bool valid() const {
    return the_predicates.valid();
  }

  CUTLASS_DEVICE
  void wait_for_current_k_tile() {
    if (!params_.tile_k) {
      return;
    }
    wait_for_k_tile(coord_offset_.contiguous() / params_.tile_k);
  }

 private:
  CUTLASS_DEVICE
  void wait_for_k_tile(int k_tile) {
    if (!params_.ready_flags || k_tile <= waited_k_tile_ || k_tile >= max_k_tiles_) {
      return;
    }

    if ((thread_idx_ & 31) == 0) {
      int slot = k_tile % params_.slot_count;
      int target = k_tile + 1;
      Atomic *ready_ptr = reinterpret_cast<Atomic *>(params_.ready_flags + slot);
      int observed = ready_ptr->load(cuda::std::memory_order_acquire);
      while (observed < target) {
        ready_ptr->wait(observed, cuda::std::memory_order_acquire);
        observed = ready_ptr->load(cuda::std::memory_order_acquire);
      }
    }
    __syncwarp();
    waited_k_tile_ = k_tile;
  }
};

template <typename Shape_, typename Element_, typename ThreadMap_, typename AccessType_>
class RingPanelTileAccessIteratorPitchLinearKInStrided {
 public:
  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::PitchLinear;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = typename Layout::TensorCoord;
  using Pointer = Element *;
  using NonConstPointer = typename platform::remove_const<Element>::type *;

  using UnderlyingPredicates = PredicatedTileAccessIteratorPredicates<
      Shape, Element, Layout, 0, ThreadMap, AccessType>;

  using Mask = typename UnderlyingPredicates::Mask;

  static int const kAccessesPerVector =
      ThreadMap::kElementsPerAccess / AccessType::kElements;

  struct Params {
    LongIndex physical_stride = 0;
    int tile_k = 0;
    int slot_count = 0;
    LongIndex slot_stride = 0;
    int *ready_flags = nullptr;

    CUTLASS_HOST_DEVICE
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(
        LongIndex physical_stride_,
        int tile_k_,
        int slot_count_,
        LongIndex slot_stride_,
        int *ready_flags_)
        : physical_stride(physical_stride_),
          tile_k(tile_k_),
          slot_count(slot_count_),
          slot_stride(slot_stride_),
          ready_flags(ready_flags_) {}
  };

 private:
  using BytePointer = char *;
  using Atomic = cuda::atomic<int, cuda::thread_scope_system>;

  UnderlyingPredicates the_predicates;
  Params params_;
  BytePointer pointer_ = nullptr;
  TensorCoord coord_offset_;
  bool is_residue_tile_ = true;
  int thread_idx_ = 0;
  int waited_k_tile_ = -1;
  int max_k_tiles_ = 0;

 public:
  CUTLASS_HOST_DEVICE
  RingPanelTileAccessIteratorPitchLinearKInStrided() = default;

  CUTLASS_HOST_DEVICE
  RingPanelTileAccessIteratorPitchLinearKInStrided(
      Params const &params,
      Pointer pointer,
      TensorCoord extent,
      int thread_id,
      TensorCoord const &threadblock_offset)
      : the_predicates(extent),
        params_(params),
        pointer_(reinterpret_cast<BytePointer>(
            const_cast<NonConstPointer>(pointer))),
        coord_offset_(threadblock_offset),
        thread_idx_(thread_id),
        max_k_tiles_(params.tile_k ? (extent.strided() + params.tile_k - 1) / params.tile_k
                                   : 0) {
    the_predicates.set_predicates(thread_id, threadblock_offset);
    coord_offset_ = the_predicates.thread_offset_;
    wait_for_k_tile(params.tile_k ? threadblock_offset.strided() / params.tile_k
                                  : 0);
  }

  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    the_predicates.set_iteration_index(index);
  }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex) {}

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    if (is_residue_tile_) {
      the_predicates.thread_offset_ += the_predicates.residue_offset_;
      the_predicates.compute_predicates_(the_predicates.extent_, true);

      coord_offset_.contiguous() =
          the_predicates.thread_offset_.contiguous() +
          Shape::kContiguous * (tile_offset.contiguous() - 1);
      coord_offset_.strided() = the_predicates.thread_offset_.strided() +
                                Shape::kStrided * tile_offset.strided();
    } else {
      coord_offset_.contiguous() +=
          Shape::kContiguous * tile_offset.contiguous();
      coord_offset_.strided() += Shape::kStrided * tile_offset.strided();
    }

    is_residue_tile_ = false;

    if (tile_offset.strided() > 0) {
      int next_k_tile = coord_offset_.strided() / params_.tile_k;
      wait_for_k_tile(next_k_tile);
    }
  }

  CUTLASS_DEVICE
  AccessType *get() const {
    if (!the_predicates.valid()) {
      return nullptr;
    }

    Index coord_contig = coord_offset_.contiguous() +
                         the_predicates.iteration_contiguous_ *
                             ThreadMap::Delta::kContiguous +
                         the_predicates.iteration_vector_ * AccessType::kElements;
    Index coord_strided = coord_offset_.strided() +
                          the_predicates.iteration_strided_ *
                              ThreadMap::Delta::kStrided;

    int k_tile = coord_strided / params_.tile_k;
    int inner_k = coord_strided % params_.tile_k;

    LongIndex offset = LongIndex(k_tile % params_.slot_count) * params_.slot_stride +
                       LongIndex(inner_k) * params_.physical_stride +
                       coord_contig;

    return reinterpret_cast<AccessType *>(pointer_ + OffsetBytes<Element>(offset));
  }

  CUTLASS_HOST_DEVICE
  RingPanelTileAccessIteratorPitchLinearKInStrided &operator++() {
    ++the_predicates.iteration_vector_;
    if (the_predicates.iteration_vector_ < kAccessesPerVector) {
      return *this;
    }

    the_predicates.iteration_vector_ = 0;
    ++the_predicates.iteration_contiguous_;
    if (the_predicates.iteration_contiguous_ < ThreadMap::Iterations::kContiguous) {
      return *this;
    }

    the_predicates.iteration_contiguous_ = 0;
    ++the_predicates.iteration_strided_;
    if (the_predicates.iteration_strided_ < ThreadMap::Iterations::kStrided) {
      return *this;
    }

    the_predicates.iteration_strided_ = 0;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  RingPanelTileAccessIteratorPitchLinearKInStrided operator++(int) {
    RingPanelTileAccessIteratorPitchLinearKInStrided self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    the_predicates.clear_mask(enable);
  }

  CUTLASS_HOST_DEVICE
  void enable_mask() {
    the_predicates.enable_mask();
  }

  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    the_predicates.set_mask(mask);
  }

  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    the_predicates.get_mask(mask);
  }

  CUTLASS_HOST_DEVICE
  bool valid() const {
    return the_predicates.valid();
  }

  CUTLASS_DEVICE
  void wait_for_current_k_tile() {
    if (!params_.tile_k) {
      return;
    }
    wait_for_k_tile(coord_offset_.strided() / params_.tile_k);
  }

 private:
  CUTLASS_DEVICE
  void wait_for_k_tile(int k_tile) {
    if (!params_.ready_flags || k_tile <= waited_k_tile_ || k_tile >= max_k_tiles_) {
      return;
    }

    if ((thread_idx_ & 31) == 0) {
      int slot = k_tile % params_.slot_count;
      int target = k_tile + 1;
      Atomic *ready_ptr = reinterpret_cast<Atomic *>(params_.ready_flags + slot);
      int observed = ready_ptr->load(cuda::std::memory_order_acquire);
      while (observed < target) {
        ready_ptr->wait(observed, cuda::std::memory_order_acquire);
        observed = ready_ptr->load(cuda::std::memory_order_acquire);
      }
    }
    __syncwarp();
    waited_k_tile_ = k_tile;
  }
};

}  // namespace detail

template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_, bool Gather, typename PermuteLayout>
class PredicatedTileAccessIterator<Shape_, Element_, layout::RowMajorRingBuffer,
                                   AdvanceRank, ThreadMap_, AccessType_, Gather,
                                   PermuteLayout> {
 public:
  static_assert(!Gather, "Ring-buffer iterator does not support gather.");
  static_assert(platform::is_same<PermuteLayout, layout::NoPermute>::value,
                "Ring-buffer iterator does not support an additional permute.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorRingBuffer;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using TensorCoord = MatrixCoord;
  using Pointer = Element *;

  using UnderlyingIterator = detail::RingPanelTileAccessIteratorPitchLinear<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element, ThreadMap,
      AccessType>;

  using Index = typename UnderlyingIterator::Index;
  using LongIndex = typename UnderlyingIterator::LongIndex;
  using Mask = typename UnderlyingIterator::Mask;
  using TensorRef = cutlass::TensorRef<Element, Layout>;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  class Params {
   private:
    friend PredicatedTileAccessIterator;
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout.stride(0),
                  layout.tile_k,
                  layout.slot_count,
                  layout.slot_stride,
                  layout.ready_flags) {}
  };

 private:
  UnderlyingIterator iterator_;

 public:
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator() = default;

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator(
      Params const &params,
      Pointer pointer,
      TensorCoord extent,
      int thread_id,
      TensorCoord const &threadblock_offset,
      int const * = nullptr)
      : iterator_(params.params_,
                  pointer,
                  layout::PitchLinearCoord(extent.column(), extent.row()),
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.column(),
                                           threadblock_offset.row())) {}

  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
  }

  CUTLASS_DEVICE
  AccessType *get() const {
    return iterator_.get();
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator operator++(int) {
    PredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  CUTLASS_HOST_DEVICE
  bool valid() const { return iterator_.valid(); }

  CUTLASS_DEVICE
  void wait_for_current_k_tile() { iterator_.wait_for_current_k_tile(); }
};

template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_, bool Gather, typename PermuteLayout>
class PredicatedTileAccessIterator<Shape_, Element_, layout::ColumnMajorRingBuffer,
                                   AdvanceRank, ThreadMap_, AccessType_, Gather,
                                   PermuteLayout> {
 public:
  static_assert(!Gather, "Ring-buffer iterator does not support gather.");
  static_assert(platform::is_same<PermuteLayout, layout::NoPermute>::value,
                "Ring-buffer iterator does not support an additional permute.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorRingBuffer;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using TensorCoord = MatrixCoord;
  using Pointer = Element *;

  using UnderlyingIterator = detail::RingPanelTileAccessIteratorPitchLinear<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element, ThreadMap,
      AccessType>;

  using Index = typename UnderlyingIterator::Index;
  using LongIndex = typename UnderlyingIterator::LongIndex;
  using Mask = typename UnderlyingIterator::Mask;
  using TensorRef = cutlass::TensorRef<Element, Layout>;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  class Params {
   private:
    friend PredicatedTileAccessIterator;
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout.stride(0),
                  layout.tile_k,
                  layout.slot_count,
                  layout.slot_stride,
                  layout.ready_flags) {}
  };

 private:
  UnderlyingIterator iterator_;

 public:
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator() = default;

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator(
      Params const &params,
      Pointer pointer,
      TensorCoord extent,
      int thread_id,
      TensorCoord const &threadblock_offset,
      int const * = nullptr)
      : iterator_(params.params_,
                  pointer,
                  layout::PitchLinearCoord(extent.row(), extent.column()),
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.row(),
                                           threadblock_offset.column())) {}

  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
  }

  CUTLASS_DEVICE
  AccessType *get() const {
    return iterator_.get();
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator operator++(int) {
    PredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  CUTLASS_HOST_DEVICE
  bool valid() const { return iterator_.valid(); }

  CUTLASS_DEVICE
  void wait_for_current_k_tile() { iterator_.wait_for_current_k_tile(); }
};

template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_, bool Gather, typename PermuteLayout>
class PredicatedTileAccessIterator<Shape_, Element_, layout::ColumnMajorRingBufferA,
                                   AdvanceRank, ThreadMap_, AccessType_, Gather,
                                   PermuteLayout> {
 public:
  static_assert(!Gather, "Ring-buffer iterator does not support gather.");
  static_assert(platform::is_same<PermuteLayout, layout::NoPermute>::value,
                "Ring-buffer iterator does not support an additional permute.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::ColumnMajorRingBufferA;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using TensorCoord = MatrixCoord;
  using Pointer = Element *;

  using UnderlyingIterator = detail::RingPanelTileAccessIteratorPitchLinearKInStrided<
      layout::PitchLinearShape<Shape::kRow, Shape::kColumn>, Element, ThreadMap,
      AccessType>;

  using Index = typename UnderlyingIterator::Index;
  using LongIndex = typename UnderlyingIterator::LongIndex;
  using Mask = typename UnderlyingIterator::Mask;
  using TensorRef = cutlass::TensorRef<Element, Layout>;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  class Params {
   private:
    friend PredicatedTileAccessIterator;
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout.stride(0),
                  layout.tile_k,
                  layout.slot_count,
                  layout.slot_stride,
                  layout.ready_flags) {}
  };

 private:
  UnderlyingIterator iterator_;

 public:
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator() = default;

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator(
      Params const &params,
      Pointer pointer,
      TensorCoord extent,
      int thread_id,
      TensorCoord const &threadblock_offset,
      int const * = nullptr)
      : iterator_(params.params_,
                  pointer,
                  layout::PitchLinearCoord(extent.row(), extent.column()),
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.row(),
                                           threadblock_offset.column())) {}

  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.row(), tile_offset.column()});
  }

  CUTLASS_DEVICE
  AccessType *get() const {
    return iterator_.get();
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator operator++(int) {
    PredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  CUTLASS_HOST_DEVICE
  bool valid() const { return iterator_.valid(); }

  CUTLASS_DEVICE
  void wait_for_current_k_tile() { iterator_.wait_for_current_k_tile(); }
};

template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_, bool Gather, typename PermuteLayout>
class PredicatedTileAccessIterator<Shape_, Element_, layout::RowMajorRingBufferB,
                                   AdvanceRank, ThreadMap_, AccessType_, Gather,
                                   PermuteLayout> {
 public:
  static_assert(!Gather, "Ring-buffer iterator does not support gather.");
  static_assert(platform::is_same<PermuteLayout, layout::NoPermute>::value,
                "Ring-buffer iterator does not support an additional permute.");

  using Shape = Shape_;
  using Element = Element_;
  using Layout = layout::RowMajorRingBufferB;
  using ThreadMap = ThreadMap_;
  using AccessType = AccessType_;
  using TensorCoord = MatrixCoord;
  using Pointer = Element *;

  using UnderlyingIterator = detail::RingPanelTileAccessIteratorPitchLinearKInStrided<
      layout::PitchLinearShape<Shape::kColumn, Shape::kRow>, Element, ThreadMap,
      AccessType>;

  using Index = typename UnderlyingIterator::Index;
  using LongIndex = typename UnderlyingIterator::LongIndex;
  using Mask = typename UnderlyingIterator::Mask;
  using TensorRef = cutlass::TensorRef<Element, Layout>;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  class Params {
   private:
    friend PredicatedTileAccessIterator;
    typename UnderlyingIterator::Params params_;

   public:
    CUTLASS_HOST_DEVICE
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout.stride(0),
                  layout.tile_k,
                  layout.slot_count,
                  layout.slot_stride,
                  layout.ready_flags) {}
  };

 private:
  UnderlyingIterator iterator_;

 public:
  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator() = default;

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator(
      Params const &params,
      Pointer pointer,
      TensorCoord extent,
      int thread_id,
      TensorCoord const &threadblock_offset,
      int const * = nullptr)
      : iterator_(params.params_,
                  pointer,
                  layout::PitchLinearCoord(extent.column(), extent.row()),
                  thread_id,
                  layout::PitchLinearCoord(threadblock_offset.column(),
                                           threadblock_offset.row())) {}

  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    iterator_.add_tile_offset({tile_offset.column(), tile_offset.row()});
  }

  CUTLASS_DEVICE
  AccessType *get() const {
    return iterator_.get();
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  CUTLASS_HOST_DEVICE
  PredicatedTileAccessIterator operator++(int) {
    PredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { iterator_.clear_mask(enable); }

  CUTLASS_HOST_DEVICE
  void enable_mask() { iterator_.enable_mask(); }

  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { iterator_.set_mask(mask); }

  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) { iterator_.get_mask(mask); }

  CUTLASS_HOST_DEVICE
  bool valid() const { return iterator_.valid(); }

  CUTLASS_DEVICE
  void wait_for_current_k_tile() { iterator_.wait_for_current_k_tile(); }
};

}  // namespace threadblock
}  // namespace transform
}  // namespace cutlass

namespace cutlass {
namespace gemm {
namespace threadblock {

template <typename Shape_, typename IteratorA_, typename SmemIteratorA_,
          cutlass::arch::CacheOperation::Kind CacheOpA, typename IteratorB_,
          typename SmemIteratorB_,
          cutlass::arch::CacheOperation::Kind CacheOpB, typename ElementC_,
          typename LayoutC_, typename Policy_, int Stages,
          SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
          typename Enable = bool>
class MmaMultistageHostPrefetch
    : public MmaBase<Shape_, Policy_, Stages> {
 public:
  using Base = MmaBase<Shape_, Policy_, Stages>;
  using Shape = Shape_;
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = Policy_;
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  using FragmentC = typename Policy::Operator::FragmentC;
  using Operator = typename Policy::Operator;
  using ArchTag = arch::Sm80;

  struct Detail {
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;
    static int const kStages = Stages;
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) /
        Base::kWarpGemmIterations;
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) /
        Base::kWarpGemmIterations;
    static bool const kStagedAccumulation =
        arch::detail::UseStagedAccumulation<Operator>::value;
  };

 private:
  struct PipeState {
    using WarpLoadedFragmentA = typename Operator::FragmentA;
    using WarpLoadedFragmentB = typename Operator::FragmentB;
    using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
    using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

    FragmentC tmp_accum_;
    WarpLoadedFragmentA warp_loaded_frag_A_[2];
    WarpTransformedFragmentA warp_transformed_frag_A_[2];
    WarpLoadedFragmentB warp_loaded_frag_B_[2];
    WarpTransformedFragmentB warp_transformed_frag_B_[2];
  };

  Operator warp_mma_;
  SmemIteratorA smem_iterator_A_;
  SmemIteratorB smem_iterator_B_;
  int smem_write_stage_idx_;
  int smem_read_stage_idx_;
  int thread_idx_;

 public:
  CUTLASS_DEVICE
  MmaMultistageHostPrefetch(typename Base::SharedStorage &shared_storage,
                            int thread_idx, int warp_idx, int lane_idx)
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
        smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
        smem_write_stage_idx_(0),
        smem_read_stage_idx_(0),
        thread_idx_(thread_idx) {
    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  CUTLASS_DEVICE
  void advance_smem_read_stage() {
    ++smem_read_stage_idx_;
    if (smem_read_stage_idx_ == Base::kStages) {
      this->warp_tile_iterator_A_.add_tile_offset(
          {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset(
          {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      smem_read_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  void advance_smem_write_stage(IteratorA &iterator_A, IteratorB &iterator_B) {
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});

    smem_iterator_A_.add_tile_offset({0, 1});
    smem_iterator_B_.add_tile_offset({1, 0});

    ++smem_write_stage_idx_;
    if (smem_write_stage_idx_ == Base::kStages) {
      smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
      smem_write_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B,
                              int group_start_A = 0, int group_start_B = 0) {
    iterator_A.set_iteration_index(group_start_A * IteratorA::kAccessesPerVector);
    smem_iterator_A_.set_iteration_index(group_start_A);

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(smem_iterator_A_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess /
                              IteratorA::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_A.get();
          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          }
          ++iterator_A;
        }
        ++smem_iterator_A_;
      }
    }

    iterator_B.set_iteration_index(group_start_B * IteratorB::kAccessesPerVector);
    smem_iterator_B_.set_iteration_index(group_start_B);

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(smem_iterator_B_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess /
                              IteratorB::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B.get();
          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
          }
          ++iterator_B;
        }
        ++smem_iterator_B_;
      }
    }
  }

  CUTLASS_DEVICE
  void prologue(IteratorA &iterator_A, IteratorB &iterator_B,
                int &gemm_k_iterations) {
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1; ++stage, --gemm_k_iterations) {
      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B.clear_mask(gemm_k_iterations == 0);

      iterator_A.set_iteration_index(0);
      smem_iterator_A_.set_iteration_index(0);

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(smem_iterator_A_.get());
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorA::Element>::value *
              IteratorA::ThreadMap::kElementsPerAccess /
              IteratorA::kAccessesPerVector / 8;
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
              dst_ptr + v, iterator_A.get(), iterator_A.valid());
          ++iterator_A;
        }
        ++smem_iterator_A_;
      }

      iterator_B.set_iteration_index(0);
      smem_iterator_B_.set_iteration_index(0);

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(smem_iterator_B_.get());
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB::Element>::value *
              IteratorB::ThreadMap::kElementsPerAccess /
              IteratorB::kAccessesPerVector / 8;
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr + v, iterator_B.get(), iterator_B.valid());
          ++iterator_B;
        }
        ++smem_iterator_B_;
      }

      advance_smem_write_stage(iterator_A, iterator_B);
      cutlass::arch::cp_async_fence();
    }
  }

  CUTLASS_DEVICE
  void gmem_wait() {
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();
  }

  CUTLASS_DEVICE
  void mac_loop_iter(PipeState &pipe_state, FragmentC &accum,
                     IteratorA &iterator_A, IteratorB &iterator_B,
                     int &gemm_k_iterations) {
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
      this->warp_tile_iterator_A_.set_kgroup_index(
          (warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_A_.load(
          pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_A_;

      this->warp_tile_iterator_B_.set_kgroup_index(
          (warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_B_.load(
          pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_B_;

      if (warp_mma_k > 0) {
        warp_mma_.transform(
            pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
            pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
            pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
            pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
      }

      if (Detail::kStagedAccumulation) {
        warp_mma_(pipe_state.tmp_accum_,
                  pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                  pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                  pipe_state.tmp_accum_);
        if (warp_mma_k == 0) {
          plus<FragmentC> plus_accum;
          accum = plus_accum(accum, pipe_state.tmp_accum_);
          pipe_state.tmp_accum_.clear();
        }
      } else {
        warp_mma_(accum,
                  pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                  pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                  accum);
      }

      if (warp_mma_k < Base::kWarpGemmIterations - 1) {
        copy_tiles_and_advance(iterator_A, iterator_B,
                               warp_mma_k * Detail::kAccessesPerGroupA,
                               warp_mma_k * Detail::kAccessesPerGroupB);
      }

      if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
        copy_tiles_and_advance(iterator_A, iterator_B,
                               (warp_mma_k + 1) * Detail::kAccessesPerGroupA,
                               (warp_mma_k + 1) * Detail::kAccessesPerGroupB);

        cutlass::arch::cp_async_fence();
        gmem_wait();

        advance_smem_write_stage(iterator_A, iterator_B);
        advance_smem_read_stage();

        --gemm_k_iterations;
        iterator_A.clear_mask(gemm_k_iterations == 0);
        iterator_B.clear_mask(gemm_k_iterations == 0);
      }

      if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
        warp_mma_.transform(
            pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
            pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
            pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
            pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      }
    }
  }

  CUTLASS_DEVICE
  void gemm_iters(int gemm_k_iterations, FragmentC &accum, IteratorA &iterator_A,
                  IteratorB &iterator_B) {
    PipeState pipe_state;

    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++this->warp_tile_iterator_A_;

    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++this->warp_tile_iterator_B_;

    warp_mma_.transform(pipe_state.warp_transformed_frag_A_[0],
                        pipe_state.warp_transformed_frag_B_[0],
                        pipe_state.warp_loaded_frag_A_[0],
                        pipe_state.warp_loaded_frag_B_[0]);

    if (Detail::kStagedAccumulation) {
      pipe_state.tmp_accum_.clear();
    }

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
      mac_loop_iter(pipe_state, accum, iterator_A, iterator_B, gemm_k_iterations);
    }

    if (Detail::kStagedAccumulation) {
      plus<FragmentC> plus_accum;
      accum = plus_accum(accum, pipe_state.tmp_accum_);
    }

    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
  }

  CUTLASS_DEVICE
  void operator()(int gemm_k_iterations, FragmentC &accum, IteratorA iterator_A,
                  IteratorB iterator_B, FragmentC const &src_accum) {
    prologue(iterator_A, iterator_B, gemm_k_iterations);
    gmem_wait();

    accum = src_accum;
    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B);
  }
};

template <typename ElementA, int kAlignmentA, typename ElementB, int kAlignmentB,
          typename ElementAccumulator, typename LayoutC, typename ArchTag,
          typename ThreadblockShape, typename WarpShape, typename InstructionShape,
          int Stages, typename Operator, SharedMemoryClearOption SharedMemoryClear>
struct DefaultMma<ElementA, layout::RowMajorRingBuffer, kAlignmentA, ElementB,
                  layout::ColumnMajorRingBuffer, kAlignmentB, ElementAccumulator,
                  LayoutC, arch::OpClassTensorOp, ArchTag, ThreadblockShape,
                  WarpShape, InstructionShape, Stages, Operator, false,
                  SharedMemoryClear, false, false, layout::NoPermute,
                  layout::NoPermute> {
  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value ||
                    platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
                "tensor-op epilogue must be row major");

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA,
      layout::RowMajor, ElementB, layout::ColumnMajor, ElementAccumulator,
      LayoutC, arch::OpClassTensorOp, Stages, Operator, false, CacheOpA,
      CacheOpB>;

  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
  using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, layout::RowMajorRingBuffer, 1, ThreadMapA, AccessTypeA>;

  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
  using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB, layout::ColumnMajorRingBuffer, 0, ThreadMapB, AccessTypeB>;

  using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistageHostPrefetch<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, LayoutC,
      typename MmaCore::MmaPolicy, Stages, SharedMemoryClear>;
};

template <typename Shape_, typename WarpShape_, typename InstructionShape_,
          typename ElementA_, typename ElementB_, typename ElementC_,
          typename LayoutC_, int Stages, typename Operator_,
          cutlass::arch::CacheOperation::Kind CacheOpA,
          cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::RowMajorRingBuffer, ElementB_,
                      layout::ColumnMajorRingBuffer, ElementC_, LayoutC_,
                      arch::OpClassTensorOp, Stages, Operator_, false, CacheOpA,
                      CacheOpB>
    : DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                     layout::RowMajor, ElementB_, layout::ColumnMajor, ElementC_,
                     LayoutC_, arch::OpClassTensorOp, Stages, Operator_, false,
                     CacheOpA, CacheOpB> {};

template <typename ElementA, int kAlignmentA, typename ElementB, int kAlignmentB,
          typename ElementAccumulator, typename LayoutC, typename ArchTag,
          typename ThreadblockShape, typename WarpShape, typename InstructionShape,
          int Stages, typename Operator, SharedMemoryClearOption SharedMemoryClear>
struct DefaultMma<ElementA, layout::ColumnMajorRingBufferA, kAlignmentA, ElementB,
                  layout::RowMajorRingBufferB, kAlignmentB, ElementAccumulator,
                  LayoutC, arch::OpClassTensorOp, ArchTag, ThreadblockShape,
                  WarpShape, InstructionShape, Stages, Operator, false,
                  SharedMemoryClear, false, false, layout::NoPermute,
                  layout::NoPermute> {
  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value ||
                    platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
                "tensor-op epilogue must be row major");

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA,
      layout::ColumnMajor, ElementB, layout::RowMajor, ElementAccumulator,
      LayoutC, arch::OpClassTensorOp, Stages, Operator, false, CacheOpA,
      CacheOpB>;

  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
  using IteratorA = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
      ElementA, layout::ColumnMajorRingBufferA, 1, ThreadMapA, AccessTypeA>;

  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
  using IteratorB = cutlass::transform::threadblock::PredicatedTileAccessIterator<
      cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
      ElementB, layout::RowMajorRingBufferB, 0, ThreadMapB, AccessTypeB>;

  using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistageHostPrefetch<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, LayoutC,
      typename MmaCore::MmaPolicy, Stages, SharedMemoryClear>;
};

template <typename Shape_, typename WarpShape_, typename InstructionShape_,
          typename ElementA_, typename ElementB_, typename ElementC_,
          typename LayoutC_, int Stages, typename Operator_,
          cutlass::arch::CacheOperation::Kind CacheOpA,
          cutlass::arch::CacheOperation::Kind CacheOpB>
struct DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                      layout::ColumnMajorRingBufferA, ElementB_,
                      layout::RowMajorRingBufferB, ElementC_, LayoutC_,
                      arch::OpClassTensorOp, Stages, Operator_, false, CacheOpA,
                      CacheOpB>
    : DefaultMmaCore<Shape_, WarpShape_, InstructionShape_, ElementA_,
                     layout::ColumnMajor, ElementB_, layout::RowMajor, ElementC_,
                     LayoutC_, arch::OpClassTensorOp, Stages, Operator_, false,
                     CacheOpA, CacheOpB> {};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
