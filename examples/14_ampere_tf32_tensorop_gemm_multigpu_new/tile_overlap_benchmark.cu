/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Benchmark: compute-vs-communication overlap feasibility at single C-tile granularity.
 *
 * Measures two quantities for tile-config=12 (TB 128x128x16, Warp 64x64x16):
 *
 *   [COMM]  P2P fetch time: one block (128 threads) pulls one A row-strip (128x4096 floats)
 *           + one B column-strip (4096x128 floats) from GPU-0 using cutlass::arch::cp_async_zfill
 *           (remote global -> shared -> local GPU-1 HBM).
 *
 *   [COMP]  Compute time: CUTLASS GemmUniversal with M=128, N=128, K=4096 (exactly one 128x128
 *           output tile) using tile-config=12 on pre-fetched local GPU-1 data.
 *
 * If COMM >> COMP, a single C tile's computation cannot hide its own communication latency.
 *
 * Usage:
 *   ./tile_overlap_benchmark [--storage-device=0] [--compute-device=1] [--iterations=100]
 *
 * K is fixed at 4096 to match the default problem in the main example.
 **************************************************************************************************/

#include <iostream>
#include <iomanip>
#include <cstdint>

#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/command_line.h"

#include "helper.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
// Tile-config=12: TB(128,128,16), Warp(64,64,16), Inst(16,8,8), NumStages=4
/////////////////////////////////////////////////////////////////////////////////////////////////

using ElementA           = float;
using ElementB           = float;
using ElementC           = float;
using ElementAccumulator = float;
using ElementEpilogue    = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using ShapeTB   = cutlass::gemm::GemmShape<128, 128, 16>;
using ShapeWarp = cutlass::gemm::GemmShape<64,  64,  16>;
using ShapeMMA  = cutlass::gemm::GemmShape<16,  8,   8>;
constexpr int kNumStages = 4;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC,
    128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccumulator,
    ElementEpilogue>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;

using Gemm = cutlass::gemm::device::GemmUniversal<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ShapeTB, ShapeWarp, ShapeMMA,
    EpilogueOp,
    SwizzleThreadBlock,
    kNumStages>;

/////////////////////////////////////////////////////////////////////////////////////////////////
// P2P fetch kernel: 128 threads + cp_async_zfill<float4>(16B)
//
// This mirrors CUTLASS's async tile copy style:
//   remote global (GPU-0) --cp.async--> shared memory --store--> local global (GPU-1)
//
// Data movement per benchmark iteration:
//   A strip: ShapeTB::kM * K floats  (one full row of A tiles)
//   B strip: K * ShapeTB::kN floats  (one full column of B tiles)
/////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int kFetchThreads = 128;
constexpr int kAccessBytes = 16;
constexpr int kAccessesPerThread = 4;
constexpr int kStageAccesses = kFetchThreads * kAccessesPerThread;
constexpr int kPipelineStages = 2;
constexpr size_t kFetchSharedBytes =
        size_t(kPipelineStages) * size_t(kStageAccesses) * sizeof(float4);

CUTLASS_DEVICE
void cp_async_fetch_stream(
        float4 const* __restrict__ remote,
        float4* __restrict__ local,
        int n_float4,
        float4* __restrict__ shared) {
    auto load_stage = [&](int stage, int tile_idx) {
        int tile_base = tile_idx * kStageAccesses;
        float4* stage_ptr = shared + stage * kStageAccesses;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kAccessesPerThread; ++i) {
            int lane_offset = threadIdx.x + i * blockDim.x;
            int access_idx = tile_base + lane_offset;
            bool valid = (access_idx < n_float4);
            cutlass::arch::cp_async_zfill<kAccessBytes>(
                    stage_ptr + lane_offset,
                    remote + (valid ? access_idx : 0),
                    valid);
        }

        cutlass::arch::cp_async_fence();
    };

    auto store_stage = [&](int stage, int tile_idx) {
        int tile_base = tile_idx * kStageAccesses;
        float4 const* stage_ptr = shared + stage * kStageAccesses;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kAccessesPerThread; ++i) {
            int lane_offset = threadIdx.x + i * blockDim.x;
            int access_idx = tile_base + lane_offset;
            if (access_idx < n_float4) {
                local[access_idx] = stage_ptr[lane_offset];
            }
        }
    };

    int tile_count = (n_float4 + kStageAccesses - 1) / kStageAccesses;
    if (tile_count == 0) {
        return;
    }

    int stage = 0;
    load_stage(stage, 0);

    for (int tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
        int next_tile = tile_idx + 1;
        int next_stage = stage ^ 1;

        if (next_tile < tile_count) {
            load_stage(next_stage, next_tile);
            cutlass::arch::cp_async_wait<1>();
        } else {
            cutlass::arch::cp_async_wait<0>();
        }

        __syncthreads();
        store_stage(stage, tile_idx);
        __syncthreads();

        stage = next_stage;
    }
}

__global__ void fetch_strip_kernel_cp_async(
        float4 const* __restrict__ remote_A,
        float4 const* __restrict__ remote_B,
        float4* __restrict__ local_A,
        float4* __restrict__ local_B,
        int n_float4_A,
        int n_float4_B) {
    extern __shared__ float4 shared[];

    cp_async_fetch_stream(remote_A, local_A, n_float4_A, shared);
    __syncthreads();

    cp_async_fetch_stream(remote_B, local_B, n_float4_B, shared);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers
/////////////////////////////////////////////////////////////////////////////////////////////////

// Returns elapsed milliseconds between two recorded events.
float event_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Main
/////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **argv) {
    cutlass::CommandLine cmd(argc, argv);

    int storage_device = 0;
    int compute_device = 1;
    int iterations     = 100;

    cmd.get_cmd_line_argument("storage-device", storage_device);
    cmd.get_cmd_line_argument("compute-device", compute_device);
    cmd.get_cmd_line_argument("iterations",     iterations);

    if (cmd.check_cmd_line_flag("help")) {
        std::cout
            << "tile_overlap_benchmark\n\n"
            << "  --storage-device=<int>  GPU holding A/B data  (default: 0)\n"
            << "  --compute-device=<int>  GPU running kernels   (default: 1)\n"
            << "  --iterations=<int>      Timed iterations      (default: 100)\n\n"
            << "Reports the P2P fetch time and compute time for one 128x128xK=4096 C tile.\n";
        return 0;
    }

    // -------------------------------------------------------------------------
    // Problem dimensions (one C tile worth of computation)
    // -------------------------------------------------------------------------
    const int M = ShapeTB::kM;   // 128
    const int N = ShapeTB::kN;   // 128
    const int K = 4096;

    const int n_float_A  = M * K;                       // 524 288 floats = 2 MB
    const int n_float_B  = K * N;                       // 524 288 floats = 2 MB
    const int n_float4_A = n_float_A / 4;               // 131 072 float4s
    const int n_float4_B = n_float_B / 4;               // 131 072 float4s
    const size_t bytes_A = n_float_A * sizeof(float);
    const size_t bytes_B = n_float_B * sizeof(float);

    std::cout << "============================================================\n";
    std::cout << "  Tile-config=12 : TB(128,128,16) Warp(64,64,16) K=" << K << "\n";
    std::cout << "  A strip  : " << M << " x " << K << " floats = "
              << bytes_A / 1024 << " KB\n";
    std::cout << "  B strip  : " << K << " x " << N << " floats = "
              << bytes_B / 1024 << " KB\n";
    std::cout << "  Total P2P: " << (bytes_A + bytes_B) / 1024 << " KB\n";
    std::cout << "  Iterations: " << iterations << "\n";
    std::cout << "============================================================\n\n";

    // -------------------------------------------------------------------------
    // GPU setup
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaSetDevice(storage_device));

    float *d_A_remote = nullptr, *d_B_remote = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_remote, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B_remote, bytes_B));

    // Fill with 1s so results are deterministic
    CUDA_CHECK(cudaMemset(d_A_remote, 0x3f, bytes_A));  // ~0.98f
    CUDA_CHECK(cudaMemset(d_B_remote, 0x3f, bytes_B));

    CUDA_CHECK(cudaSetDevice(compute_device));

    // Enable P2P if GPUs differ
    if (compute_device != storage_device) {
        int can_access = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, compute_device, storage_device));
        if (!can_access) {
            std::cerr << "ERROR: GPU " << compute_device
                      << " cannot access GPU " << storage_device << " via P2P.\n";
            return 1;
        }
        cudaError_t e = cudaDeviceEnablePeerAccess(storage_device, 0);
        if (e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
    }

    // Local copies on compute GPU (destination of P2P fetch)
    float *d_A_local = nullptr, *d_B_local = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_local, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B_local, bytes_B));

    // C output
    float *d_C = nullptr, *d_D = nullptr;
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_D, 0, M * N * sizeof(float)));

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // =========================================================================
    // [1] P2P Fetch Benchmark
    //     One block, 128 threads — same thread count as tile-config=12 GEMM block.
    //     Uses cp_async_zfill to fetch A strip + B strip from remote GPU to local GPU.
    // =========================================================================

    // Warm-up (fills cache lines, establishes NVLink connection)
    for (int i = 0; i < 5; ++i) {
        fetch_strip_kernel_cp_async<<<1, kFetchThreads, kFetchSharedBytes>>>(
            reinterpret_cast<float4 const*>(d_A_remote),
            reinterpret_cast<float4 const*>(d_B_remote),
            reinterpret_cast<float4*>(d_A_local),
            reinterpret_cast<float4*>(d_B_local),
            n_float4_A, n_float4_B);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < iterations; ++i) {
        fetch_strip_kernel_cp_async<<<1, kFetchThreads, kFetchSharedBytes>>>(
            reinterpret_cast<float4 const*>(d_A_remote),
            reinterpret_cast<float4 const*>(d_B_remote),
            reinterpret_cast<float4*>(d_A_local),
            reinterpret_cast<float4*>(d_B_local),
            n_float4_A, n_float4_B);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float total_fetch_ms = event_ms(ev_start, ev_stop);
    float avg_fetch_us   = total_fetch_ms / iterations * 1000.f;
    double fetch_bw_GBs  =
        double(bytes_A + bytes_B) / (double(avg_fetch_us) * 1e-6) / 1e9;

    // =========================================================================
    // [2] Compute Benchmark
    //     CUTLASS GemmUniversal M=128, N=128, K=4096, tile-config=12, local data.
    //     A is on local GPU memory (d_A_local filled by P2P fetch above).
    // =========================================================================

    Gemm gemm_op;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K},
        /*split_k_slices=*/1,
        {ElementEpilogue(1.f), ElementEpilogue(0.f)},
        d_A_local,   // A on local GPU (already fetched)
        d_B_local,   // B on local GPU
        d_C,
        d_D,
        /*batch_stride_A=*/int64_t(M) * K,
        /*batch_stride_B=*/int64_t(K) * N,
        /*batch_stride_C=*/int64_t(M) * N,
        /*batch_stride_D=*/int64_t(M) * N,
        /*lda=*/K,
        /*ldb=*/K,
        /*ldc=*/N,
        /*ldd=*/N,
        /*avail_sms=*/-1};

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM cannot implement: " << cutlass::cutlassGetStatusString(status) << "\n";
        return 1;
    }

    size_t workspace_bytes = Gemm::get_workspace_size(args);
    void *d_workspace = nullptr;
    if (workspace_bytes > 0) CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));

    status = gemm_op.initialize(args, d_workspace);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM init failed: " << cutlass::cutlassGetStatusString(status) << "\n";
        return 1;
    }

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "GEMM run failed\n"; return 1;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int i = 0; i < iterations; ++i) {
        gemm_op();
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float total_compute_ms = event_ms(ev_start, ev_stop);
    float avg_compute_us   = total_compute_ms / iterations * 1000.f;
    double tflops = 2.0 * M * N * K / (avg_compute_us * 1e-6) / 1e12;

    // =========================================================================
    // Report
    // =========================================================================
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "---------- P2P Fetch (A strip + B strip, 1 block / 128 threads) ----------\n";
    std::cout << "  Transfer size    : " << (bytes_A + bytes_B) / 1024 << " KB  ("
              << bytes_A / 1024 << " KB A + " << bytes_B / 1024 << " KB B)\n";
    std::cout << "  Avg time         : " << avg_fetch_us << " us\n";
    std::cout << "  Effective BW     : " << fetch_bw_GBs << " GB/s\n\n";

    std::cout << "---------- Compute  (128x128xK=4096 GEMM, tile-config=12) ----------------\n";
    std::cout << "  Avg time         : " << avg_compute_us << " us\n";
    std::cout << "  TFLOPs           : " << tflops << "\n\n";

    std::cout << "---------- Overlap Feasibility -------------------------------------------\n";
    float ratio = avg_fetch_us / avg_compute_us;
    std::cout << "  fetch_time / compute_time = " << ratio << "\n";
    if (ratio <= 1.1f) {
        std::cout << "  => Compute and communication are comparable: overlap is FEASIBLE.\n";
    } else {
        std::cout << "  => Communication is " << ratio << "x slower than compute.\n";
        std::cout << "     A single C-tile cannot hide its own P2P fetch latency.\n";
        std::cout << "     Consider K-slice-stationary order to amortize comm over "
                  << (int)ratio << "+ tiles.\n";
    }
    std::cout << "===========================================================================\n";

    // Cleanup
    CUDA_CHECK(cudaSetDevice(storage_device));
    cudaFree(d_A_remote);
    cudaFree(d_B_remote);

    CUDA_CHECK(cudaSetDevice(compute_device));
    cudaFree(d_A_local);
    cudaFree(d_B_local);
    cudaFree(d_C);
    cudaFree(d_D);
    if (d_workspace) cudaFree(d_workspace);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return 0;
}
