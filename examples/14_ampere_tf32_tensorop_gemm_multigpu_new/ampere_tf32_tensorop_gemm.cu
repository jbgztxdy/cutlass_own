/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/**
Please check example 07 and 08 for the basics of tensor op gemm kernels.  On NVIDIA Ampere
architecture, most concept still holds.  The two main differences are

1. NVIDIA Ampere architecture introduces a new series of tensor core instructions (see 
   include/cutlass/arch/mma_sm80.h) which are more efficient on Ampere.

2. NVIDIA Ampere architecture uses cp_async() to build multistage software pipeline to better hide
   latency (see include/cutlass/gemm/threadblock/mma_multistage.h)

Moreover, NVIDIA Ampere architecture starts supporting tfloat32 (see include/cutlass/tfloat32.h)
data types in tensor cores.  One big advantage is that we can load in fp32 data and convert them
implicitly to tf32 inside the GEMM kernel which means no change is needed to accelerate traditional
fp32 data by using NVIDIA Ampere architecture.
*/

#include <cstring>
#include <iostream>
#include <cuda.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "helper.h"
#include "host_prefetch_gemm.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct Options {

  bool help;

  cutlass::gemm::GemmCoord problem_size;
  int batch_count;
  float alpha;
  float beta;

  bool reference_check;
  int iterations;
  int tile_config;
  int storage_device;   // GPU to store data
  int compute_device;  // GPU to run kernel
  
  Options():
    help(false),
    problem_size({5120, 4096, 4096}),
    batch_count(1),
    reference_check(true),
    iterations(20),
    tile_config(0),
    storage_device(0),
    compute_device(1),
    alpha(1),
    beta() { }

  bool valid() {
    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    cmd.get_cmd_line_argument("m", problem_size.m());
    cmd.get_cmd_line_argument("n", problem_size.n());
    cmd.get_cmd_line_argument("k", problem_size.k());

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("tile-config", tile_config);
    cmd.get_cmd_line_argument("storage-device", storage_device);
    cmd.get_cmd_line_argument("compute-device", compute_device);

    // Support both "--key=value" and "--key value" forms.
    for (int i = 1; i + 1 < argc; ++i) {
      if (!std::strcmp(args[i], "--tile-config")) {
        tile_config = std::atoi(args[i + 1]);
      } else if (!std::strcmp(args[i], "--storage-device")) {
        storage_device = std::atoi(args[i + 1]);
      } else if (!std::strcmp(args[i], "--compute-device")) {
        compute_device = std::atoi(args[i + 1]);
      }
    }

  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "14_ampere_tf32_tensorop_gemm example\n\n"
      << "  This example uses the CUTLASS Library to execute TF32 tensorop GEMM computations.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --tile-config=<int>         Tile config id (SM80 TF32 catalog):\n"
      << "                               0  -> TB(32,32,16),   Warp(16,16,16),  Inst(16,8,8)\n"
      << "                               1  -> TB(64,64,16),   Warp(16,32,16),  Inst(16,8,8)\n"
      << "                               2  -> TB(64,64,16),   Warp(32,32,16),  Inst(16,8,8)\n"
      << "                               3  -> TB(64,64,32),   Warp(32,32,32),  Inst(16,8,8)\n"
      << "                               4  -> TB(64,128,16),  Warp(32,64,16),  Inst(16,8,8)\n"
      << "                               5  -> TB(64,128,32),  Warp(32,64,32),  Inst(16,8,8)\n"
      << "                               6  -> TB(64,256,16),  Warp(64,64,16),  Inst(16,8,8)\n"
      << "                               7  -> TB(64,256,32),  Warp(64,64,32),  Inst(16,8,8)\n"
      << "                               8  -> TB(128,64,16),  Warp(64,32,16),  Inst(16,8,8)\n"
      << "                               9  -> TB(128,64,32),  Warp(32,64,32),  Inst(16,8,8)\n"
      << "                               10 -> TB(128,64,32),  Warp(64,32,32),  Inst(16,8,8)\n"
      << "                               11 -> TB(128,128,16), Warp(32,64,16),  Inst(16,8,8)\n"
      << "                               12 -> TB(128,128,16), Warp(64,64,16),  Inst(16,8,8)\n"
      << "                               13 -> TB(128,128,32), Warp(64,64,32),  Inst(16,8,8)\n"
      << "                               14 -> TB(128,256,16), Warp(64,64,16),  Inst(16,8,8)\n"
      << "                               15 -> TB(128,256,16), Warp(64,128,16), Inst(16,8,8)\n"
      << "                               16 -> TB(128,256,32), Warp(64,64,32),  Inst(16,8,8)\n"
      << "                               17 -> TB(256,64,16),  Warp(64,64,16),  Inst(16,8,8)\n"
      << "                               18 -> TB(256,64,32),  Warp(64,64,32),  Inst(16,8,8)\n"
      << "                               19 -> TB(256,128,16), Warp(64,64,16),  Inst(16,8,8)\n"
      << "                               20 -> TB(256,128,16), Warp(128,64,16), Inst(16,8,8)\n"
      << "                               21 -> TB(256,128,32), Warp(64,64,32),  Inst(16,8,8)\n"
      << "                               22 -> TB(256,256,16), Warp(64,128,16), Inst(16,8,8)\n"
      << "                               23 -> TB(256,256,32), Warp(128,128,32), Inst(16,8,8)\n"
      << "                               24 -> TB(512,128,16), Warp(128,64,16), Inst(16,8,8)\n"
      << "                               25 -> TB(512,256,16), Warp(128,128,16), Inst(16,8,8)\n"
      << "                               26 -> TB(512,512,16), Warp(128,128,16), Inst(16,8,8)\n"
      << "  --storage-device=<int>       GPU device to store data (default: 0)\n"
      << "  --compute-device=<int>       GPU device to run kernel (default: 1)\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/14_ampere_tf32_tensorop_gemm_multigpu/14_ampere_tf32_tensorop_gemm_multigpu --m=1024 --n=512 --k=1024 \\\n"
      << "     --storage-device=0 --compute-device=1\n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = problem_size.product() * batch_count;
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = float;                        // <- data type of elements in input matrix A
using ElementInputB = float;                        // <- data type of elements in input matrix B
using ElementOutput = float;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputAHost = cutlass::layout::RowMajor;
using LayoutInputBHost = cutlass::layout::ColumnMajor;
using LayoutInputAKernel = cutlass::layout::RowMajorRingBuffer;
using LayoutInputBKernel = cutlass::layout::ColumnMajorRingBuffer;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// Tile-shape catalog (SM80 TF32 + Inst 16x8x8), sourced from CUTLASS unit-test combinations.
// Threadblock shapes
using ShapeTB_32_32_16 = cutlass::gemm::GemmShape<32, 32, 16>;
using ShapeTB_64_64_16 = cutlass::gemm::GemmShape<64, 64, 16>;
using ShapeTB_64_64_32 = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeTB_64_128_16 = cutlass::gemm::GemmShape<64, 128, 16>;
using ShapeTB_64_128_32 = cutlass::gemm::GemmShape<64, 128, 32>;
using ShapeTB_64_256_16 = cutlass::gemm::GemmShape<64, 256, 16>;
using ShapeTB_64_256_32 = cutlass::gemm::GemmShape<64, 256, 32>;
using ShapeTB_128_64_16 = cutlass::gemm::GemmShape<128, 64, 16>;
using ShapeTB_128_64_32 = cutlass::gemm::GemmShape<128, 64, 32>;
using ShapeTB_128_128_16 = cutlass::gemm::GemmShape<128, 128, 16>;
using ShapeTB_128_128_32 = cutlass::gemm::GemmShape<128, 128, 32>;
using ShapeTB_128_256_16 = cutlass::gemm::GemmShape<128, 256, 16>;
using ShapeTB_128_256_32 = cutlass::gemm::GemmShape<128, 256, 32>;
using ShapeTB_256_64_16 = cutlass::gemm::GemmShape<256, 64, 16>;
using ShapeTB_256_64_32 = cutlass::gemm::GemmShape<256, 64, 32>;
using ShapeTB_256_128_16 = cutlass::gemm::GemmShape<256, 128, 16>;
using ShapeTB_256_128_32 = cutlass::gemm::GemmShape<256, 128, 32>;
using ShapeTB_256_256_16 = cutlass::gemm::GemmShape<256, 256, 16>;
using ShapeTB_256_256_32 = cutlass::gemm::GemmShape<256, 256, 32>;
using ShapeTB_512_128_16 = cutlass::gemm::GemmShape<512, 128, 16>;
using ShapeTB_512_256_16 = cutlass::gemm::GemmShape<512, 256, 16>;
using ShapeTB_512_512_16 = cutlass::gemm::GemmShape<512, 512, 16>;

// Warp shapes
using ShapeWarp_16_16_16 = cutlass::gemm::GemmShape<16, 16, 16>;
using ShapeWarp_16_32_16 = cutlass::gemm::GemmShape<16, 32, 16>;
using ShapeWarp_32_32_16 = cutlass::gemm::GemmShape<32, 32, 16>;
using ShapeWarp_32_32_32 = cutlass::gemm::GemmShape<32, 32, 32>;
using ShapeWarp_32_64_16 = cutlass::gemm::GemmShape<32, 64, 16>;
using ShapeWarp_32_64_32 = cutlass::gemm::GemmShape<32, 64, 32>;
using ShapeWarp_64_32_16 = cutlass::gemm::GemmShape<64, 32, 16>;
using ShapeWarp_64_32_32 = cutlass::gemm::GemmShape<64, 32, 32>;
using ShapeWarp_64_64_16 = cutlass::gemm::GemmShape<64, 64, 16>;
using ShapeWarp_64_64_32 = cutlass::gemm::GemmShape<64, 64, 32>;
using ShapeWarp_64_64_64 = cutlass::gemm::GemmShape<64, 64, 64>;
using ShapeWarp_64_128_16 = cutlass::gemm::GemmShape<64, 128, 16>;
using ShapeWarp_128_64_16 = cutlass::gemm::GemmShape<128, 64, 16>;
using ShapeWarp_128_128_16 = cutlass::gemm::GemmShape<128, 128, 16>;
using ShapeWarp_128_128_32 = cutlass::gemm::GemmShape<128, 128, 32>;

#define TILE_CONFIG_LIST(X) \
  X(0, ShapeTB_32_32_16, ShapeWarp_16_16_16) \
  X(1, ShapeTB_64_64_16, ShapeWarp_16_32_16) \
  X(2, ShapeTB_64_64_16, ShapeWarp_32_32_16) \
  X(3, ShapeTB_64_64_32, ShapeWarp_32_32_32) \
  X(4, ShapeTB_64_128_16, ShapeWarp_32_64_16) \
  X(5, ShapeTB_64_128_32, ShapeWarp_32_64_32) \
  X(6, ShapeTB_64_256_16, ShapeWarp_64_64_16) \
  X(7, ShapeTB_64_256_32, ShapeWarp_64_64_32) \
  X(8, ShapeTB_128_64_16, ShapeWarp_64_32_16) \
  X(9, ShapeTB_128_64_32, ShapeWarp_32_64_32) \
  X(10, ShapeTB_128_64_32, ShapeWarp_64_32_32) \
  X(11, ShapeTB_128_128_16, ShapeWarp_32_64_16) \
  X(12, ShapeTB_128_128_16, ShapeWarp_64_64_16) \
  X(13, ShapeTB_128_128_32, ShapeWarp_64_64_32) \
  X(14, ShapeTB_128_256_16, ShapeWarp_64_64_16) \
  X(15, ShapeTB_128_256_16, ShapeWarp_64_128_16) \
  X(16, ShapeTB_128_256_32, ShapeWarp_64_64_32) \
  X(17, ShapeTB_256_64_16, ShapeWarp_64_64_16) \
  X(18, ShapeTB_256_64_32, ShapeWarp_64_64_32) \
  X(19, ShapeTB_256_128_16, ShapeWarp_64_64_16) \
  X(20, ShapeTB_256_128_16, ShapeWarp_128_64_16) \
  X(21, ShapeTB_256_128_32, ShapeWarp_64_64_32) \
  X(22, ShapeTB_256_256_16, ShapeWarp_64_128_16) \
  X(23, ShapeTB_256_256_32, ShapeWarp_128_128_32) \
  X(24, ShapeTB_512_128_16, ShapeWarp_128_64_16) \
  X(25, ShapeTB_512_256_16, ShapeWarp_128_128_16) \
  X(26, ShapeTB_512_512_16, ShapeWarp_128_128_16)
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 4;

template <typename ShapeThreadblock_, typename ShapeWarp_>
using GemmKernel = cutlass::gemm::device::Gemm<ElementInputA,
                                               LayoutInputAKernel,
                                               ElementInputB,
                                               LayoutInputBKernel,
                                               ElementOutput,
                                               LayoutOutput,
                                               ElementAccumulator,
                                               MMAOp,
                                               SmArch,
                                               ShapeThreadblock_,
                                               ShapeWarp_,
                                               ShapeMMAOp,
                                               EpilogueOp,
                                               SwizzleThreadBlock,
                                               NumStages>;

#define DECLARE_GEMM_CFG(ID, TB, WARP) using GemmCfg##ID = GemmKernel<TB, WARP>;
TILE_CONFIG_LIST(DECLARE_GEMM_CFG)
#undef DECLARE_GEMM_CFG

#define CU_CHECK(status)                                                        \
  {                                                                             \
    CUresult error = status;                                                    \
    if (error != CUDA_SUCCESS) {                                                \
      const char *error_string = nullptr;                                       \
      cuGetErrorString(error, &error_string);                                   \
      std::cerr << "Got bad driver status: "                                    \
                << (error_string ? error_string : "unknown")                    \
                << " at line: " << __LINE__ << std::endl;                       \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  }

namespace {
inline void stream_write_value32(cudaStream_t stream, int *device_ptr, int value) {
  CU_CHECK(cuStreamWriteValue32(
      reinterpret_cast<CUstream>(stream),
      reinterpret_cast<CUdeviceptr>(device_ptr),
      static_cast<cuuint32_t>(value), 0));
}

template <typename ElementA_, typename ElementB_>
struct PrefetchWorkspace {
  int slot_count = 0;
  int tile_k = 0;
  size_t slot_elements_a = 0;
  size_t slot_elements_b = 0;
  cutlass::device_memory::allocation<ElementA_> staged_a;
  cutlass::device_memory::allocation<ElementB_> staged_b;
  cutlass::device_memory::allocation<int> ready_flags;

  PrefetchWorkspace()
      : staged_a(0),
        staged_b(0),
        ready_flags(0) {}

  PrefetchWorkspace(PrefetchWorkspace const &) = delete;
  PrefetchWorkspace &operator=(PrefetchWorkspace const &) = delete;
};

inline int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

template <typename ThreadblockShape>
int grid_tile_count(cutlass::gemm::GemmCoord const &problem_size) {
  SwizzleThreadBlock swizzle;
  cutlass::gemm::GemmCoord tiled_shape = swizzle.get_tiled_shape(
      problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      1);
  return tiled_shape.m() * tiled_shape.n() * tiled_shape.k();
}

template <typename ElementA_, typename ElementB_>
void allocate_prefetch_workspace(
    PrefetchWorkspace<ElementA_, ElementB_> &workspace,
    cutlass::gemm::GemmCoord const &problem_size,
    int tile_k,
    int slot_count) {
  workspace.tile_k = tile_k;
  workspace.slot_count = slot_count;
  workspace.slot_elements_a = size_t(problem_size.m()) * size_t(tile_k);
  workspace.slot_elements_b = size_t(problem_size.n()) * size_t(tile_k);

  workspace.staged_a.reset(workspace.slot_elements_a * size_t(slot_count));
  workspace.staged_b.reset(workspace.slot_elements_b * size_t(slot_count));
  workspace.ready_flags.reset(slot_count);
}

template <typename ElementA_, typename ElementB_>
void reset_prefetch_workspace(
    PrefetchWorkspace<ElementA_, ElementB_> &workspace,
    cudaStream_t copy_stream) {
  CUDA_CHECK(cudaMemsetAsync(workspace.ready_flags.get(), 0,
                             sizeof(int) * workspace.slot_count, copy_stream));
}

template <typename ElementA_, typename ElementB_>
void enqueue_k_tile_copy(
    PrefetchWorkspace<ElementA_, ElementB_> &workspace,
    cutlass::gemm::GemmCoord const &problem_size,
    cutlass::HostTensor<ElementA_, LayoutInputAHost> &tensor_a,
    cutlass::HostTensor<ElementB_, LayoutInputBHost> &tensor_b,
    int k_tile_idx,
    cudaStream_t copy_stream) {
  int slot = k_tile_idx % workspace.slot_count;
  int tile_k = workspace.tile_k;
  int valid_k = std::min(tile_k, problem_size.k() - k_tile_idx * tile_k);
  size_t width_bytes_a = size_t(valid_k) * sizeof(ElementA_);
  size_t width_bytes_b = size_t(valid_k) * sizeof(ElementB_);

  ElementA_ *dst_a = workspace.staged_a.get() +
                     size_t(slot) * workspace.slot_elements_a;
  ElementA_ const *src_a = tensor_a.device_data() + size_t(k_tile_idx) * tile_k;
  CUDA_CHECK(cudaMemcpy2DAsync(
      dst_a, size_t(tile_k) * sizeof(ElementA_), src_a,
      size_t(problem_size.k()) * sizeof(ElementA_), width_bytes_a,
      problem_size.m(), cudaMemcpyDeviceToDevice, copy_stream));

  ElementB_ *dst_b = workspace.staged_b.get() +
                     size_t(slot) * workspace.slot_elements_b;
  ElementB_ const *src_b = tensor_b.device_data() + size_t(k_tile_idx) * tile_k;
  CUDA_CHECK(cudaMemcpy2DAsync(
      dst_b, size_t(tile_k) * sizeof(ElementB_), src_b,
      size_t(problem_size.k()) * sizeof(ElementB_), width_bytes_b,
      problem_size.n(), cudaMemcpyDeviceToDevice, copy_stream));
}

template <typename ElementA_, typename ElementB_>
void publish_ready_flag(
    PrefetchWorkspace<ElementA_, ElementB_> &workspace,
    int slot,
    int ready_token,
    cudaStream_t copy_stream) {
  stream_write_value32(copy_stream, workspace.ready_flags.get() + slot,
                       ready_token);
}

template <typename Gemm, typename ThreadblockShape>
cutlass::Status initialize_gemm(
    Gemm &gemm_op,
    cutlass::gemm::GemmCoord const &problem_size,
    PrefetchWorkspace<ElementInputA, ElementInputB> &prefetch_workspace,
    cutlass::HostTensor<ElementOutput, LayoutOutput> &tensor_c,
    cutlass::HostTensor<ElementOutput, LayoutOutput> &tensor_d,
    ElementComputeEpilogue alpha,
    ElementComputeEpilogue beta,
    cutlass::device_memory::allocation<uint8_t> &workspace) {
  typename Gemm::Arguments arguments{
      problem_size,
      typename Gemm::TensorRefA(
          prefetch_workspace.staged_a.get(),
          LayoutInputAKernel(
              prefetch_workspace.tile_k, prefetch_workspace.slot_count,
              cutlass::layout::RowMajorRingBuffer::LongIndex(
                  prefetch_workspace.slot_elements_a),
              prefetch_workspace.ready_flags.get())),
      typename Gemm::TensorRefB(
          prefetch_workspace.staged_b.get(),
          LayoutInputBKernel(
              prefetch_workspace.tile_k, prefetch_workspace.slot_count,
              cutlass::layout::ColumnMajorRingBuffer::LongIndex(
                  prefetch_workspace.slot_elements_b),
              prefetch_workspace.ready_flags.get())),
      tensor_c.device_ref(),
      tensor_d.device_ref(),
      {alpha, beta},
      1};

  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  workspace.reset(Gemm::get_workspace_size(arguments));

  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);
  return status;
}

template <typename Gemm, typename ThreadblockShape>
cutlass::Status run_gemm_iteration(
    Gemm &gemm_op,
    cutlass::gemm::GemmCoord const &problem_size,
    cutlass::HostTensor<ElementInputA, LayoutInputAHost> &tensor_a,
    cutlass::HostTensor<ElementInputB, LayoutInputBHost> &tensor_b,
    PrefetchWorkspace<ElementInputA, ElementInputB> &prefetch_workspace,
    int tile_k_count,
    cudaStream_t compute_stream,
    cudaStream_t copy_stream,
    cudaEvent_t start_event,
    cudaEvent_t stop_event) {
  reset_prefetch_workspace(prefetch_workspace, copy_stream);

  for (int k_tile_idx = 0; k_tile_idx < tile_k_count; ++k_tile_idx) {
    enqueue_k_tile_copy(prefetch_workspace, problem_size, tensor_a, tensor_b,
                        k_tile_idx, copy_stream);
    int slot = k_tile_idx;
    publish_ready_flag(prefetch_workspace, slot, k_tile_idx + 1,
                       copy_stream);
  }

  CUDA_CHECK(cudaEventRecord(start_event, compute_stream));
  cutlass::Status status = gemm_op(compute_stream);
  CUTLASS_CHECK(status);
  CUDA_CHECK(cudaEventRecord(stop_event, compute_stream));

  return status;
}

template <typename Gemm, typename ThreadblockShape>
Result run_configured_gemm(
    Gemm &gemm_op,
    Options const &options,
    cutlass::gemm::GemmCoord const &problem_size,
    cutlass::HostTensor<ElementInputA, LayoutInputAHost> &tensor_a,
    cutlass::HostTensor<ElementInputB, LayoutInputBHost> &tensor_b,
    cutlass::HostTensor<ElementOutput, LayoutOutput> &tensor_c,
    cutlass::HostTensor<ElementOutput, LayoutOutput> &tensor_d) {
  Result result;
  cutlass::Status status = cutlass::Status::kSuccess;

  int tile_k_count = ceil_div(problem_size.k(), ThreadblockShape::kK);
  int slot_count = tile_k_count;

  PrefetchWorkspace<ElementInputA, ElementInputB> prefetch_workspace;
  allocate_prefetch_workspace(prefetch_workspace, problem_size,
                              ThreadblockShape::kK, slot_count);

  cutlass::device_memory::allocation<uint8_t> gemm_workspace(1);
  status = initialize_gemm<Gemm, ThreadblockShape>(
      gemm_op, problem_size, prefetch_workspace, tensor_c, tensor_d,
      ElementComputeEpilogue(options.alpha),
      ElementComputeEpilogue(options.beta), gemm_workspace);
  CUTLASS_CHECK(status);

  cudaStream_t compute_stream = nullptr;
  cudaStream_t copy_stream = nullptr;
  cudaEvent_t start_event = nullptr;
  cudaEvent_t stop_event = nullptr;

  CUDA_CHECK(cudaStreamCreate(&compute_stream));
  CUDA_CHECK(cudaStreamCreate(&copy_stream));
  CUDA_CHECK(cudaEventCreate(&start_event));
  CUDA_CHECK(cudaEventCreate(&stop_event));

  double total_runtime_ms = 0.0;

  for (int iter = 0; iter < options.iterations; ++iter) {
    status = run_gemm_iteration<Gemm, ThreadblockShape>(
        gemm_op, problem_size, tensor_a, tensor_b, prefetch_workspace,
        tile_k_count, compute_stream, copy_stream, start_event, stop_event);
    CUTLASS_CHECK(status);
    CUDA_CHECK(cudaEventSynchronize(stop_event));

    float runtime_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&runtime_ms, start_event, stop_event));
    total_runtime_ms += double(runtime_ms);
  }

  CUDA_CHECK(cudaEventDestroy(stop_event));
  CUDA_CHECK(cudaEventDestroy(start_event));
  CUDA_CHECK(cudaStreamDestroy(copy_stream));
  CUDA_CHECK(cudaStreamDestroy(compute_stream));

  result.runtime_ms = total_runtime_ms / double(options.iterations);
  result.gflops = options.gflops(result.runtime_ms / 1000.0);
  result.status = status;
  return result;
}

}  // namespace

int run(Options &options) {
  cutlass::gemm::GemmCoord problem_size = options.problem_size;

  cudaError_t cuda_status = cudaSetDevice(options.storage_device);
  if (cuda_status != cudaSuccess) {
    std::cerr << "Failed to set storage device to " << options.storage_device
              << std::endl;
    return -1;
  }

  cutlass::HostTensor<ElementInputA, LayoutInputAHost> tensor_a(problem_size.mk());
  cutlass::HostTensor<ElementInputB, LayoutInputBHost> tensor_b(problem_size.kn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(problem_size.mn());
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(problem_size.mn());

  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(), 1, ElementInputA(4), ElementInputA(-4), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(), 1, ElementInputB(4), ElementInputB(-4), 0);
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(), 1, ElementOutput(4), ElementOutput(-4), 0);
  cutlass::reference::host::TensorFill(tensor_d.host_view());
  cutlass::reference::host::TensorFill(tensor_ref_d.host_view());

  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  if (options.compute_device != options.storage_device) {
    cuda_status = cudaSetDevice(options.compute_device);
    if (cuda_status != cudaSuccess) {
      std::cerr << "Failed to set compute device to " << options.compute_device
                << std::endl;
      return -1;
    }

    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, options.compute_device,
                            options.storage_device);
    if (can_access) {
      cuda_status = cudaDeviceEnablePeerAccess(options.storage_device, 0);
      if (cuda_status != cudaSuccess &&
          cuda_status != cudaErrorPeerAccessAlreadyEnabled) {
        std::cerr << "Failed to enable peer access: "
                  << cudaGetErrorString(cuda_status) << std::endl;
      }
    } else {
      std::cerr << "Warning: Peer access not available between GPU "
                << options.compute_device << " and GPU " << options.storage_device
                << std::endl;
    }
  } else {
    CUDA_CHECK(cudaSetDevice(options.compute_device));
  }

  Result result;
  int selected_config = options.tile_config;

  #define DECLARE_GEMM_INSTANCE(ID, TB, WARP) GemmCfg##ID gemm_cfg_##ID;
  TILE_CONFIG_LIST(DECLARE_GEMM_INSTANCE)
  #undef DECLARE_GEMM_INSTANCE

  switch (selected_config) {
  #define RUN_CASE(ID, TB, WARP)                                             \
    case ID:                                                                  \
      result = run_configured_gemm<GemmCfg##ID, TB>(                          \
          gemm_cfg_##ID, options, problem_size, tensor_a, tensor_b, tensor_c, \
          tensor_d);                                                          \
      break;
  TILE_CONFIG_LIST(RUN_CASE)
  #undef RUN_CASE
    default:
      std::cerr << "Unsupported --tile-config value: " << selected_config
                << ". Supported values are 0..26." << std::endl;
      return -1;
  }

  cutlass::reference::device::Gemm<ElementInputA, LayoutInputAHost, ElementInputB,
                                   LayoutInputBHost, ElementOutput, LayoutOutput,
                                   ElementComputeEpilogue, ElementComputeEpilogue>
      gemm_device;

  gemm_device(problem_size, ElementComputeEpilogue(options.alpha),
              tensor_a.device_ref(), tensor_b.device_ref(),
              ElementComputeEpilogue(options.beta), tensor_c.device_ref(),
              tensor_ref_d.device_ref());

  cudaDeviceSynchronize();

  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  bool passed = cutlass::reference::host::TensorEquals(tensor_d.host_view(),
                                                       tensor_ref_d.host_view());

  if (passed) {
    std::cout << "Runtime: " << result.runtime_ms << " ms" << std::endl;
    std::cout << " GFLOPs: " << result.gflops << std::endl;
  }

  std::cout << (passed ? "Passed" : "Failed") << std::endl;
  return (passed ? 0 : -1);
}

int main(int argc, const char **argv) {
  
  bool notSupported = false;

  // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
  // in CUDA 11.0. 
  //
  // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  // Check for multiple GPUs
  int device_count;
  error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount() failed: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  Options options;
  options.parse(argc, argv);

  // Validate device indices
  if (options.storage_device < 0 || options.storage_device >= device_count ||
      options.compute_device < 0 || options.compute_device >= device_count) {
    std::cerr << "Invalid device indices. Device count: " << device_count << std::endl;
    std::cerr << "Storage device: " << options.storage_device << std::endl;
    std::cerr << "Compute device: " << options.compute_device << std::endl;
    return -1;
  }

  if (notSupported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  printf("%d x %d x %d TF32 tensor op Matrix Multiply\n", \
    options.problem_size.m(), options.problem_size.n(), options.problem_size.k());

  printf("Storage GPU: %d, Compute GPU: %d\n", 
    options.storage_device, options.compute_device);

  if (options.storage_device != options.compute_device) {
    printf("Note: Host prefetches K-panels from storage GPU to compute GPU HBM before each stage\n");
  }

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  return run(options);
}
