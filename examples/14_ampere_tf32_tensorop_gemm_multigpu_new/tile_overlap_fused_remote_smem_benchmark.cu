/***************************************************************************************************
 * Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Benchmark: fused remote-to-shared + compute at one C-tile granularity.
 *
 * Compared to tile_overlap_benchmark.cu (two-kernel flow), this benchmark runs a single kernel:
 *
 *   remote global (GPU-0) --cp.async--> local shared memory (GPU-1) --direct compute--> C tile
 *
 * It reports two timing components measured inside the kernel with clock64:
 *   [LOAD]    remote -> smem cycles
 *   [COMPUTE] math cycles using smem-resident A/B tiles
 *
 * Problem shape is fixed to tile-config=12-compatible dimensions for one output tile:
 *   M=128, N=128, K=4096
 *
 * Usage:
 *   ./tile_overlap_fused_remote_smem_benchmark
 *     [--storage-device=0] [--compute-device=1] [--iterations=100]
 **************************************************************************************************/

#include <cstdint>
#include <iomanip>
#include <iostream>

#include <mma.h>

#include "cutlass/arch/memory_sm80.h"
#include "cutlass/cutlass.h"
#include "cutlass/util/command_line.h"

#include "helper.h"

namespace {

// Keep the same tile shape as tile-config=12.
constexpr int kTileM = 128;
constexpr int kTileN = 128;
constexpr int kTileK = 16;
constexpr int kKTotal = 4096;

constexpr int kThreads = 128;
constexpr int kAccessBytes = 16;

constexpr int kATileElems = kTileM * kTileK;  // 2048 floats
constexpr int kBTileElems = kTileN * kTileK;  // 2048 floats; stored as [n][kk]
constexpr int kCTileElems = kTileM * kTileN;  // 16384 floats

constexpr int kAFloat4PerTile = kATileElems / 4;  // 512
constexpr int kBFloat4PerTile = kBTileElems / 4;  // 512

// wmma tile shape for TF32 on sm80: m16n16k8
// WarpShape 64×64×16 decomposes into 4×4×2 = 32 wmma.mma_sync calls per warp per k-tile,
// matching the 4×8×2=64 mma.sync count of the original config-12 warp (2 mma per wmma call).
using namespace nvcuda;
constexpr int kWmmaM = 16, kWmmaN = 16, kWmmaK = 8;
constexpr int kWarpTilesM = 64 / kWmmaM;  // 4
constexpr int kWarpTilesN = 64 / kWmmaN;  // 4
constexpr int kWarpTilesK = kTileK / kWmmaK;  // 2

// Shared memory layout in fused kernel:
//   [C_tile(128x128)] [A_k_tile(128x16)] [B_k_tile(128x16)]
constexpr size_t kFusedSharedBytes =
    size_t(kCTileElems + kATileElems + kBTileElems) * sizeof(float);

__global__ void fused_remote_smem_compute_kernel(
    float const* __restrict__ remote_A,       // row-major [M, K]
    float const* __restrict__ remote_B_col,   // column-major [K, N], packed as [n][k]
    float* __restrict__ checksum_out,
    unsigned long long* __restrict__ load_cycles_out,
    unsigned long long* __restrict__ compute_cycles_out,
    int K,
    int repeats) {

  extern __shared__ float shared[];
  float* C_smem = shared;
  float* A_smem = C_smem + kCTileElems;
  float* B_smem = A_smem + kATileElems;

  float4* A_smem4 = reinterpret_cast<float4*>(A_smem);
  float4* B_smem4 = reinterpret_cast<float4*>(B_smem);

  unsigned long long load_cycles = 0;
  unsigned long long compute_cycles = 0;

  int warp_idx = threadIdx.x >> 5;
  int warp_m = warp_idx & 1;     // 0..1
  int warp_n = warp_idx >> 1;    // 0..1

  // wmma accumulators: warp covers 64×64 output, decomposed into 4×4 wmma m16n16 tiles.
  // Declared outside rep loop so the compiler keeps them in registers across k0 iterations.
  wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> accum[kWarpTilesM][kWarpTilesN];

  for (int rep = 0; rep < repeats; ++rep) {

    // Zero accumulators at the start of each rep.
    for (int mi = 0; mi < kWarpTilesM; ++mi)
      for (int ni = 0; ni < kWarpTilesN; ++ni)
        wmma::fill_fragment(accum[mi][ni], 0.0f);

    // C_smem is only used for the checksum at the end; no need to pre-zero it here.
    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += kTileK) {
      unsigned long long load_start = 0;

      __syncthreads();
      if (threadIdx.x == 0) {
        load_start = clock64();
      }
      __syncthreads();

      // Load A subtile [M, kTileK] as float4 via cp.async.
      for (int i = threadIdx.x; i < kAFloat4PerTile; i += blockDim.x) {
        int base = i * 4;
        int m = base / kTileK;
        int kk = base % kTileK;  // 0,4,8,12 for alignment

        float const* src = remote_A + m * K + (k0 + kk);
        cutlass::arch::cp_async_zfill<kAccessBytes>(A_smem4 + i, src, true);
      }

      // Load B subtile as [n, kTileK].
      // Since B is column-major [K, N], address is n*K + k.
      for (int i = threadIdx.x; i < kBFloat4PerTile; i += blockDim.x) {
        int base = i * 4;
        int n = base / kTileK;
        int kk = base % kTileK;  // 0,4,8,12 for alignment

        float const* src = remote_B_col + n * K + (k0 + kk);
        cutlass::arch::cp_async_zfill<kAccessBytes>(B_smem4 + i, src, true);
      }

      cutlass::arch::cp_async_fence();
      cutlass::arch::cp_async_wait<0>();
      __syncthreads();

      if (threadIdx.x == 0) {
        load_cycles += (clock64() - load_start);
      }
      __syncthreads();

      unsigned long long compute_start = 0;
      if (threadIdx.x == 0) {
        compute_start = clock64();
      }
      __syncthreads();

      // Compute: wmma API reads plain row/col-major smem correctly (no swizzle needed).
      // A_smem[m * kTileK + k]: row-major [128, kTileK], ld = kTileK.
      // B_smem[n * kTileK + k]: stored as [N, kTileK] — equivalent to col-major [K, N] with ld = kTileK.
      {
        float* A_warp = A_smem + warp_m * 64 * kTileK;  // warp's 64 rows of A
        float* B_warp = B_smem + warp_n * 64 * kTileK;  // warp's 64 cols of B

        #pragma unroll
        for (int ki = 0; ki < kWarpTilesK; ++ki) {
          #pragma unroll
          for (int mi = 0; mi < kWarpTilesM; ++mi) {
            wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                           wmma::precision::tf32, wmma::row_major> a_frag;
            // A sub-tile: rows [mi*16, mi*16+16), k-group [ki*8, ki*8+8)
            wmma::load_matrix_sync(a_frag,
                                   A_warp + mi * kWmmaM * kTileK + ki * kWmmaK,
                                   kTileK);
            #pragma unroll
            for (int ni = 0; ni < kWarpTilesN; ++ni) {
              wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                             wmma::precision::tf32, wmma::col_major> b_frag;
              // B sub-tile (col-major view): "rows" = K-group, "cols" = N-group.
              // B_warp[ni*kWmmaN * kTileK + ki*kWmmaK] = start of col-range [ni*16..] at k=ki*8.
              wmma::load_matrix_sync(b_frag,
                                     B_warp + ni * kWmmaN * kTileK + ki * kWmmaK,
                                     kTileK);
              wmma::mma_sync(accum[mi][ni], a_frag, b_frag, accum[mi][ni]);
            }
          }
        }
      }

      __syncthreads();
      if (threadIdx.x == 0) {
        compute_cycles += (clock64() - compute_start);
      }
      __syncthreads();
    }

    // Store wmma accumulators into C_smem (row-major) for checksum.
    {
      int row_off = warp_m * 64;
      int col_off = warp_n * 64;
      for (int mi = 0; mi < kWarpTilesM; ++mi) {
        for (int ni = 0; ni < kWarpTilesN; ++ni) {
          float* C_ptr = C_smem + (row_off + mi * kWmmaM) * kTileN + (col_off + ni * kWmmaN);
          wmma::store_matrix_sync(C_ptr, accum[mi][ni], kTileN, wmma::mem_row_major);
        }
      }
    }
    __syncthreads();

    // Prevent dead-code elimination and provide a simple sanity scalar.
    float local_sum = 0.0f;
    for (int idx = threadIdx.x; idx < kCTileElems; idx += blockDim.x) {
      local_sum += C_smem[idx];
    }

    __shared__ float block_sum[kThreads];
    block_sum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
      if (threadIdx.x < stride) {
        block_sum[threadIdx.x] += block_sum[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      atomicAdd(checksum_out, block_sum[0]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    *load_cycles_out = load_cycles;
    *compute_cycles_out = compute_cycles;
  }
}

double cycles_to_us(unsigned long long cycles, int clock_rate_khz) {
  double cycles_per_us = double(clock_rate_khz) / 1000.0;
  return cycles_per_us > 0.0 ? double(cycles) / cycles_per_us : 0.0;
}

}  // namespace

int main(int argc, char const** argv) {
  cutlass::CommandLine cmd(argc, argv);

  int storage_device = 0;
  int compute_device = 1;
  int iterations = 100;

  cmd.get_cmd_line_argument("storage-device", storage_device);
  cmd.get_cmd_line_argument("compute-device", compute_device);
  cmd.get_cmd_line_argument("iterations", iterations);

  if (cmd.check_cmd_line_flag("help")) {
    std::cout
        << "tile_overlap_fused_remote_smem_benchmark\n\n"
        << "  --storage-device=<int>  GPU holding remote A/B (default: 0)\n"
        << "  --compute-device=<int>  GPU running fused kernel (default: 1)\n"
        << "  --iterations=<int>      Fused repeats in timed kernel (default: 100)\n\n"
        << "Reports in-kernel [remote->smem] and [compute] times for one 128x128x4096 C tile.\n";
    return 0;
  }

  if (iterations <= 0) {
    std::cerr << "iterations must be > 0\n";
    return 1;
  }

  constexpr int M = kTileM;
  constexpr int N = kTileN;
  constexpr int K = kKTotal;

  const int n_float_A = M * K;
  const int n_float_B = K * N;
  const size_t bytes_A = size_t(n_float_A) * sizeof(float);
  const size_t bytes_B = size_t(n_float_B) * sizeof(float);

  std::cout << "============================================================\n";
  std::cout << "  Fused remote->smem->compute benchmark (config12-compatible)\n";
  std::cout << "  Tile shape     : M=" << M << " N=" << N << " K=" << K << "\n";
  std::cout << "  A strip bytes  : " << bytes_A / 1024 << " KB\n";
  std::cout << "  B strip bytes  : " << bytes_B / 1024 << " KB\n";
  std::cout << "  Per-repeat load: " << (bytes_A + bytes_B) / 1024 << " KB\n";
  std::cout << "  Threads/block  : " << kThreads << "\n";
  std::cout << "  Shared bytes   : " << kFusedSharedBytes / 1024 << " KB\n";
  std::cout << "  Iterations     : " << iterations << "\n";
  std::cout << "============================================================\n\n";

  // -------------------------------------------------------------------------
  // Allocate remote source buffers on storage GPU.
  // -------------------------------------------------------------------------
  CUDA_CHECK(cudaSetDevice(storage_device));

  float* d_A_remote = nullptr;
  float* d_B_remote = nullptr;
  CUDA_CHECK(cudaMalloc(&d_A_remote, bytes_A));
  CUDA_CHECK(cudaMalloc(&d_B_remote, bytes_B));

  // Use deterministic values.
  CUDA_CHECK(cudaMemset(d_A_remote, 0x3f, bytes_A));
  CUDA_CHECK(cudaMemset(d_B_remote, 0x3f, bytes_B));

  // -------------------------------------------------------------------------
  // Prepare compute GPU and peer access.
  // -------------------------------------------------------------------------
  CUDA_CHECK(cudaSetDevice(compute_device));

  if (compute_device != storage_device) {
    int can_access = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, compute_device, storage_device));
    if (!can_access) {
      std::cerr << "ERROR: GPU " << compute_device
                << " cannot access GPU " << storage_device << " via P2P.\n";
      return 1;
    }
    cudaError_t e = cudaDeviceEnablePeerAccess(storage_device, 0);
    if (e != cudaErrorPeerAccessAlreadyEnabled) {
      CUDA_CHECK(e);
    }
  }

  float* d_checksum = nullptr;
  unsigned long long* d_load_cycles = nullptr;
  unsigned long long* d_compute_cycles = nullptr;

  CUDA_CHECK(cudaMalloc(&d_checksum, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_load_cycles, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&d_compute_cycles, sizeof(unsigned long long)));

  CUDA_CHECK(cudaMemset(d_checksum, 0, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_load_cycles, 0, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_compute_cycles, 0, sizeof(unsigned long long)));

  int max_optin_smem = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(
      &max_optin_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, compute_device));

  if (int(kFusedSharedBytes) > max_optin_smem) {
    std::cerr << "ERROR: required shared memory " << kFusedSharedBytes
              << " exceeds opt-in max " << max_optin_smem << " on compute device.\n";
    return 1;
  }

  CUDA_CHECK(cudaFuncSetAttribute(
      fused_remote_smem_compute_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      int(kFusedSharedBytes)));

  // Prefer shared-memory-heavy carveout for this microbenchmark.
  CUDA_CHECK(cudaFuncSetAttribute(
      fused_remote_smem_compute_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout,
      100));

  // -------------------------------------------------------------------------
  // Warmup
  // -------------------------------------------------------------------------
  for (int i = 0; i < 5; ++i) {
    fused_remote_smem_compute_kernel<<<1, kThreads, kFusedSharedBytes>>>(
        d_A_remote,
        d_B_remote,
        d_checksum,
        d_load_cycles,
        d_compute_cycles,
        K,
        1);
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemset(d_checksum, 0, sizeof(float)));
  CUDA_CHECK(cudaMemset(d_load_cycles, 0, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(d_compute_cycles, 0, sizeof(unsigned long long)));

  cudaEvent_t ev_start, ev_stop;
  CUDA_CHECK(cudaEventCreate(&ev_start));
  CUDA_CHECK(cudaEventCreate(&ev_stop));

  CUDA_CHECK(cudaEventRecord(ev_start));
  fused_remote_smem_compute_kernel<<<1, kThreads, kFusedSharedBytes>>>(
      d_A_remote,
      d_B_remote,
      d_checksum,
      d_load_cycles,
      d_compute_cycles,
      K,
      iterations);
  CUDA_CHECK(cudaEventRecord(ev_stop));
  CUDA_CHECK(cudaEventSynchronize(ev_stop));

  float total_kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_kernel_ms, ev_start, ev_stop));

  unsigned long long load_cycles = 0;
  unsigned long long compute_cycles = 0;
  float checksum = 0.0f;
  CUDA_CHECK(cudaMemcpy(
      &load_cycles, d_load_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      &compute_cycles, d_compute_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&checksum, d_checksum, sizeof(float), cudaMemcpyDeviceToHost));

  int clock_rate_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, compute_device));

  double total_load_us = cycles_to_us(load_cycles, clock_rate_khz);
  double total_compute_us = cycles_to_us(compute_cycles, clock_rate_khz);
  double avg_load_us = total_load_us / double(iterations);
  double avg_compute_us = total_compute_us / double(iterations);

  double load_bw_gbs =
      double(bytes_A + bytes_B) / (avg_load_us * 1e-6) / 1e9;
  double tflops =
      2.0 * double(M) * double(N) * double(K) / (avg_compute_us * 1e-6) / 1e12;

  double ratio = avg_load_us / avg_compute_us;

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "---------- In-kernel LOAD (remote -> smem) -------------------------------\n";
  std::cout << "  Avg time         : " << avg_load_us << " us\n";
  std::cout << "  Effective BW     : " << load_bw_gbs << " GB/s\n\n";

  std::cout << "---------- In-kernel COMPUTE (using smem A/B directly) -------------------\n";
  std::cout << "  Avg time         : " << avg_compute_us << " us\n";
  std::cout << "  Equivalent TFLOPs: " << tflops << "\n\n";

  std::cout << "---------- Fused Kernel Summary -------------------------------------------\n";
  std::cout << "  Kernel wall time : " << total_kernel_ms << " ms (for " << iterations
            << " repeats)\n";
  std::cout << "  load/compute     : " << ratio << "\n";
  std::cout << "  checksum         : " << checksum << "\n";
  std::cout << "===========================================================================\n";

  CUDA_CHECK(cudaSetDevice(storage_device));
  cudaFree(d_A_remote);
  cudaFree(d_B_remote);

  CUDA_CHECK(cudaSetDevice(compute_device));
  cudaFree(d_checksum);
  cudaFree(d_load_cycles);
  cudaFree(d_compute_cycles);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  return 0;
}
