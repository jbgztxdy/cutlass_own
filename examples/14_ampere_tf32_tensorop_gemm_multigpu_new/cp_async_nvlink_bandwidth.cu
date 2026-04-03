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

/*!
  \file
  \brief Measures device-side peer-memory copy throughput when GPU1 pulls from GPU0 with cp.async.
*/

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>

#include "cutlass/arch/memory_sm80.h"
#include "cutlass/util/command_line.h"

#include "helper.h"

namespace {

constexpr int kThreads = 256;
constexpr int kAccessBytes = 16;
constexpr int kAccessesPerThread = 4;
constexpr int kStageAccesses = kThreads * kAccessesPerThread;
constexpr int kStageBytes = kStageAccesses * kAccessBytes;
constexpr int kStages = 2;

struct Options {
  bool help = false;
  int src_device = 0;
  int dst_device = 1;
  size_t bytes = size_t(512) << 20;
  int warmup = 5;
  int iterations = 30;
  int blocks_per_sm = 8;
  bool run_memcpy_peer = true;
  bool run_direct = true;
  bool run_latency = true;
  bool verify = true;
  int latency_iterations = 4096;
  size_t latency_stride_bytes = size_t(256) << 10;

  void parse(int argc, char const** argv) {
    cutlass::CommandLine cmd(argc, argv);

    help = cmd.check_cmd_line_flag("help");
    cmd.get_cmd_line_argument("src-device", src_device);
    cmd.get_cmd_line_argument("dst-device", dst_device);
    cmd.get_cmd_line_argument("bytes", bytes);
    cmd.get_cmd_line_argument("warmup", warmup);
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("blocks-per-sm", blocks_per_sm);
    cmd.get_cmd_line_argument("run-memcpy-peer", run_memcpy_peer);
    cmd.get_cmd_line_argument("run-direct", run_direct);
    cmd.get_cmd_line_argument("run-latency", run_latency);
    cmd.get_cmd_line_argument("verify", verify);
    cmd.get_cmd_line_argument("latency-iterations", latency_iterations);
    cmd.get_cmd_line_argument("latency-stride-bytes", latency_stride_bytes);
  }

  std::ostream& print_usage(std::ostream& out) const {
    out
      << "cp_async_nvlink_bandwidth\n\n"
      << "This benchmark launches a kernel on --dst-device and uses cp.async to pull data\n"
      << "from peer memory allocated on --src-device into shared memory, then stores it into\n"
      << "global memory on --dst-device.\n\n"
      << "Important: this is not a device-to-device DMA engine benchmark. It measures the path\n"
      << "where SMs on the destination GPU consume peer memory through cp.async.\n\n"
      << "Options:\n"
      << "  --help                  Print this message\n"
      << "  --src-device=<int>      Source GPU id (default: 0)\n"
      << "  --dst-device=<int>      Destination GPU id / kernel launch GPU (default: 1)\n"
      << "  --bytes=<int>           Requested transfer size in bytes (default: 536870912)\n"
      << "  --warmup=<int>          Warmup iterations per method (default: 5)\n"
      << "  --iterations=<int>      Timed iterations per method (default: 30)\n"
      << "  --blocks-per-sm=<int>   Resident blocks target per SM (default: 8)\n"
      << "  --run-memcpy-peer=<0|1> Also benchmark cudaMemcpyPeerAsync (default: 1)\n"
      << "  --run-direct=<0|1>      Also benchmark direct uint4 global loads/stores (default: 1)\n"
      << "  --run-latency=<0|1>     Also measure serialized cp.async latency (default: 1)\n"
      << "  --verify=<0|1>          Verify destination contents after each method (default: 1)\n"
      << "  --latency-iterations    Serialized latency samples (default: 4096)\n"
      << "  --latency-stride-bytes  Address stride between latency samples (default: 262144)\n\n"
      << "Example:\n"
      << "  ./examples/14_ampere_tf32_tensorop_gemm_multigpu_new/cp_async_nvlink_bandwidth "
      << "--src-device=0 --dst-device=1 --bytes=1073741824\n";
    return out;
  }
};

struct BenchmarkResult {
  std::string name;
  size_t bytes_per_iteration = 0;
  int iterations = 0;
  float runtime_ms = 0.0f;
  float total_runtime_ms = 0.0f;
  double total_bytes = 0.0;
  double gb_per_s = 0.0;
  double gib_per_s = 0.0;
  unsigned long long mismatches = 0;
  bool verified = false;
};

struct LatencyResult {
  int iterations = 0;
  size_t stride_bytes = 0;
  double average_cycles = 0.0;
  double minimum_cycles = 0.0;
  double average_ns = 0.0;
  double minimum_ns = 0.0;
};

bool is_peer_access_already_enabled(cudaError_t error) {
  return error == cudaErrorPeerAccessAlreadyEnabled;
}

void enable_peer_access_one_way(int current_device, int peer_device) {
  CUDA_CHECK(cudaSetDevice(current_device));
  cudaError_t peer_status = cudaDeviceEnablePeerAccess(peer_device, 0);
  if (peer_status != cudaSuccess && !is_peer_access_already_enabled(peer_status)) {
    CUDA_CHECK(peer_status);
  }
  if (peer_status == cudaErrorPeerAccessAlreadyEnabled) {
    CUDA_CHECK(cudaGetLastError());
  }
}

void enable_peer_access_bidirectional(int src_device, int dst_device) {
  int can_access = 0;

  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, dst_device, src_device));
  if (!can_access) {
    std::cerr << "Peer access is not available from device " << dst_device
              << " to device " << src_device << ".\n";
    std::exit(EXIT_FAILURE);
  }

  CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, src_device, dst_device));
  if (!can_access) {
    std::cerr << "Peer access is not available from device " << src_device
              << " to device " << dst_device << ".\n";
    std::exit(EXIT_FAILURE);
  }

  enable_peer_access_one_way(dst_device, src_device);
  enable_peer_access_one_way(src_device, dst_device);
}

double to_gigabytes_per_second(size_t bytes, float runtime_ms) {
  double seconds = double(runtime_ms) * 1.0e-3;
  return seconds > 0.0 ? double(bytes) / 1.0e9 / seconds : 0.0;
}

double to_gibibytes_per_second(size_t bytes, float runtime_ms) {
  double seconds = double(runtime_ms) * 1.0e-3;
  return seconds > 0.0 ? double(bytes) / double(1ull << 30) / seconds : 0.0;
}

size_t round_down_to_access_granularity(size_t bytes) {
  return bytes / kAccessBytes * kAccessBytes;
}

__global__ void initialize_source_kernel(uint4* dst, size_t access_count) {
  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(blockDim.x) * gridDim.x;

  for (; idx < access_count; idx += stride) {
    unsigned base = static_cast<unsigned>(idx * 4);
    dst[idx] = make_uint4(base + 0u, base + 1u, base + 2u, base + 3u);
  }
}

__global__ void direct_peer_copy_kernel(uint4* dst, uint4 const* src, size_t access_count) {
  size_t tile_idx = blockIdx.x;
  size_t tile_stride = gridDim.x;

  while (tile_idx * kStageAccesses < access_count) {
    size_t tile_base = tile_idx * kStageAccesses;

    #pragma unroll
    for (int i = 0; i < kAccessesPerThread; ++i) {
      size_t access_idx = tile_base + threadIdx.x + size_t(i) * blockDim.x;
      if (access_idx < access_count) {
        dst[access_idx] = src[access_idx];
      }
    }

    tile_idx += tile_stride;
  }
}

__global__ void cp_async_peer_copy_kernel(uint4* dst, uint4 const* src, size_t access_count) {
  extern __shared__ uint4 shared[];

  auto load_tile = [&](int stage, size_t tile_idx) {
    size_t tile_base = tile_idx * kStageAccesses;
    uint4* stage_ptr = shared + stage * kStageAccesses;

    #pragma unroll
    for (int i = 0; i < kAccessesPerThread; ++i) {
      int lane_offset = threadIdx.x + i * blockDim.x;
      size_t access_idx = tile_base + lane_offset;
      bool valid = access_idx < access_count;
      cutlass::arch::cp_async_zfill<kAccessBytes>(
        stage_ptr + lane_offset,
        src + (valid ? access_idx : 0),
        valid);
    }

    cutlass::arch::cp_async_fence();
  };

  auto store_tile = [&](int stage, size_t tile_idx) {
    size_t tile_base = tile_idx * kStageAccesses;
    uint4 const* stage_ptr = shared + stage * kStageAccesses;

    #pragma unroll
    for (int i = 0; i < kAccessesPerThread; ++i) {
      int lane_offset = threadIdx.x + i * blockDim.x;
      size_t access_idx = tile_base + lane_offset;
      if (access_idx < access_count) {
        dst[access_idx] = stage_ptr[lane_offset];
      }
    }
  };

  size_t tile_count = (access_count + kStageAccesses - 1) / kStageAccesses;
  size_t tile_idx = blockIdx.x;
  if (tile_idx >= tile_count) {
    return;
  }

  int stage = 0;
  load_tile(stage, tile_idx);

  while (true) {
    size_t next_tile = tile_idx + gridDim.x;
    int next_stage = stage ^ 1;

    if (next_tile < tile_count) {
      load_tile(next_stage, next_tile);
      cutlass::arch::cp_async_wait<1>();
    } else {
      cutlass::arch::cp_async_wait<0>();
    }
    __syncthreads();
    store_tile(stage, tile_idx);
    __syncthreads();

    if (next_tile >= tile_count) {
      break;
    }

    tile_idx = next_tile;
    stage = next_stage;
  }

  cutlass::arch::cp_async_wait<0>();
}

__global__ void compare_buffers_kernel(
  uint4 const* expected,
  uint4 const* actual,
  size_t access_count,
  unsigned long long* mismatches) {

  size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(blockDim.x) * gridDim.x;
  unsigned long long local_mismatches = 0;

  for (; idx < access_count; idx += stride) {
    uint4 lhs = expected[idx];
    uint4 rhs = actual[idx];
    if (lhs.x != rhs.x || lhs.y != rhs.y || lhs.z != rhs.z || lhs.w != rhs.w) {
      ++local_mismatches;
    }
  }

  if (local_mismatches) {
    atomicAdd(mismatches, local_mismatches);
  }
}

__global__ void cp_async_latency_kernel(
  uint4 const* src,
  size_t access_count,
  size_t stride_accesses,
  int iterations,
  unsigned long long* total_cycles,
  unsigned long long* min_cycles,
  uint4* sink) {

  __shared__ uint4 shared_slot[1];

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  unsigned long long local_total = 0;
  unsigned long long local_min = ~0ull;
  uint4 accum = make_uint4(0u, 0u, 0u, 0u);

  for (int iter = 0; iter < iterations; ++iter) {
    size_t access_idx = (size_t(iter) * stride_accesses) % access_count;

    unsigned long long start = clock64();
    cutlass::arch::cp_async_zfill<kAccessBytes>(&shared_slot[0], src + access_idx, true);
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
    unsigned long long stop = clock64();

    unsigned long long delta = stop - start;
    local_total += delta;
    local_min = delta < local_min ? delta : local_min;

    accum.x ^= shared_slot[0].x;
    accum.y ^= shared_slot[0].y;
    accum.z ^= shared_slot[0].z;
    accum.w ^= shared_slot[0].w;
  }

  *total_cycles = local_total;
  *min_cycles = local_min;
  *sink = accum;
}

unsigned long long verify_copy(uint4 const* src_peer, uint4 const* dst_local, size_t access_count, int device) {
  CUDA_CHECK(cudaSetDevice(device));

  unsigned long long* mismatches_device = nullptr;
  CUDA_CHECK(cudaMalloc(&mismatches_device, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(mismatches_device, 0, sizeof(unsigned long long)));

  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device));
  int blocks = std::max(1, sm_count * 4);

  compare_buffers_kernel<<<blocks, kThreads>>>(src_peer, dst_local, access_count, mismatches_device);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long mismatches_host = 0;
  CUDA_CHECK(cudaMemcpy(&mismatches_host, mismatches_device, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(mismatches_device));
  return mismatches_host;
}

template <typename Launcher>
BenchmarkResult benchmark_method(
  std::string name,
  Launcher launcher,
  cudaStream_t stream,
  size_t bytes,
  int warmup,
  int iterations,
  bool verify,
  uint4 const* src_peer,
  uint4* dst_local,
  size_t access_count,
  int device) {

  CUDA_CHECK(cudaSetDevice(device));

  for (int iter = 0; iter < warmup; ++iter) {
    launcher(stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  GpuTimer timer;
  timer.start(stream);
  for (int iter = 0; iter < iterations; ++iter) {
    launcher(stream);
  }
  timer.stop();

  BenchmarkResult result;
  result.name = std::move(name);
  result.bytes_per_iteration = bytes;
  result.iterations = std::max(1, iterations);
  result.total_runtime_ms = timer.elapsed_millis();
  result.total_bytes = double(bytes) * double(result.iterations);
  result.runtime_ms = result.total_runtime_ms / float(result.iterations);
  result.gb_per_s = to_gigabytes_per_second(bytes, result.runtime_ms);
  result.gib_per_s = to_gibibytes_per_second(bytes, result.runtime_ms);

  CUDA_CHECK(cudaMemsetAsync(dst_local, 0, bytes, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  launcher(stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  if (verify) {
    result.verified = true;
    result.mismatches = verify_copy(src_peer, dst_local, access_count, device);
  }

  return result;
}

void print_device_summary(int src_device, int dst_device) {
  cudaDeviceProp src_prop{};
  cudaDeviceProp dst_prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&src_prop, src_device));
  CUDA_CHECK(cudaGetDeviceProperties(&dst_prop, dst_device));

  int performance_rank = -1;
  CUDA_CHECK(cudaDeviceGetP2PAttribute(
    &performance_rank,
    cudaDevP2PAttrPerformanceRank,
    dst_device,
    src_device));

  std::cout << "Source GPU : " << src_device << " (" << src_prop.name << ")\n";
  std::cout << "Dest GPU   : " << dst_device << " (" << dst_prop.name << ")\n";
  std::cout << "P2P rank   : " << performance_rank
            << " (lower is usually better; confirm NVLink with nvidia-smi topo -m)\n";
}

void print_result(BenchmarkResult const& result) {
  std::cout << std::left << std::setw(18) << result.name
            << " avg " << std::setw(9) << std::fixed << std::setprecision(3) << result.runtime_ms << " ms"
            << "  " << std::setw(8) << std::setprecision(2) << result.gb_per_s << " GB/s"
            << "  " << std::setw(8) << result.gib_per_s << " GiB/s";

  if (!result.verified) {
    std::cout << "  not-verified";
  } else if (result.mismatches == 0) {
    std::cout << "  verified";
  } else {
    std::cout << "  mismatches=" << result.mismatches;
  }

  std::cout << "\n";
  std::cout << "  bytes/iter=" << result.bytes_per_iteration
            << ", iterations=" << result.iterations
            << ", total_bytes=" << std::fixed << std::setprecision(0) << result.total_bytes
            << ", total_time_ms=" << std::setprecision(3) << result.total_runtime_ms << "\n";
  std::cout << "  throughput = total_bytes / total_time"
            << " = " << std::setprecision(0) << result.total_bytes
            << " / " << std::setprecision(6) << (double(result.total_runtime_ms) * 1.0e-3)
            << " s = " << std::setprecision(6)
            << (result.total_bytes / (double(result.total_runtime_ms) * 1.0e-3))
            << " B/s"
            << " = " << std::setprecision(2) << result.gb_per_s << " GB/s"
            << " = " << result.gib_per_s << " GiB/s\n";
}

LatencyResult measure_cp_async_latency(
  uint4 const* src_peer,
  size_t access_count,
  int iterations,
  size_t stride_bytes,
  int device) {

  CUDA_CHECK(cudaSetDevice(device));

  unsigned long long* total_cycles_device = nullptr;
  unsigned long long* min_cycles_device = nullptr;
  uint4* sink_device = nullptr;

  CUDA_CHECK(cudaMalloc(&total_cycles_device, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&min_cycles_device, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(&sink_device, sizeof(uint4)));

  CUDA_CHECK(cudaMemset(total_cycles_device, 0, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(min_cycles_device, 0xff, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(sink_device, 0, sizeof(uint4)));

  size_t stride_accesses = std::max<size_t>(1, stride_bytes / sizeof(uint4));
  cp_async_latency_kernel<<<1, 1>>>(
    src_peer, access_count, stride_accesses, iterations,
    total_cycles_device, min_cycles_device, sink_device);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  unsigned long long total_cycles = 0;
  unsigned long long min_cycles = 0;
  uint4 sink_host{};
  CUDA_CHECK(cudaMemcpy(&total_cycles, total_cycles_device, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&min_cycles, min_cycles_device, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&sink_host, sink_device, sizeof(uint4), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(total_cycles_device));
  CUDA_CHECK(cudaFree(min_cycles_device));
  CUDA_CHECK(cudaFree(sink_device));

  int clock_rate_khz = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device));
  double cycles_per_ns = double(clock_rate_khz) / 1.0e6;

  LatencyResult result;
  result.iterations = iterations;
  result.stride_bytes = stride_accesses * sizeof(uint4);
  result.average_cycles = iterations > 0 ? double(total_cycles) / double(iterations) : 0.0;
  result.minimum_cycles = double(min_cycles);
  result.average_ns = cycles_per_ns > 0.0 ? result.average_cycles / cycles_per_ns : 0.0;
  result.minimum_ns = cycles_per_ns > 0.0 ? result.minimum_cycles / cycles_per_ns : 0.0;

  if ((sink_host.x | sink_host.y | sink_host.z | sink_host.w) == 0u && access_count > 0) {
    asm volatile("" : : : "memory");
  }

  return result;
}

void print_latency_result(LatencyResult const& result) {
  std::cout << "cp.async peer latency\n";
  std::cout << "  mode = serialized single-thread single-request measurement\n";
  std::cout << "  samples=" << result.iterations
            << ", stride_bytes=" << result.stride_bytes << "\n";
  std::cout << "  average latency = " << std::fixed << std::setprecision(2)
            << result.average_cycles << " cycles = "
            << result.average_ns << " ns\n";
  std::cout << "  minimum latency = " << result.minimum_cycles << " cycles = "
            << result.minimum_ns << " ns\n";
}

}  // namespace

int main(int argc, char const** argv) {
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout);
    return 0;
  }

  int device_count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    std::cerr << "This benchmark requires at least two CUDA devices.\n";
    return EXIT_FAILURE;
  }

  if (options.src_device == options.dst_device) {
    std::cerr << "--src-device and --dst-device must be different.\n";
    return EXIT_FAILURE;
  }

  if (options.src_device < 0 || options.src_device >= device_count ||
      options.dst_device < 0 || options.dst_device >= device_count) {
    std::cerr << "Requested device ids are out of range.\n";
    return EXIT_FAILURE;
  }

  size_t measured_bytes = round_down_to_access_granularity(options.bytes);
  if (measured_bytes == 0) {
    std::cerr << "--bytes must be at least " << kAccessBytes << ".\n";
    return EXIT_FAILURE;
  }

  enable_peer_access_bidirectional(options.src_device, options.dst_device);
  print_device_summary(options.src_device, options.dst_device);

  if (measured_bytes != options.bytes) {
    std::cout << "Requested " << options.bytes << " bytes, benchmarking "
              << measured_bytes << " bytes to preserve 16-byte cp.async alignment.\n";
  } else {
    std::cout << "Benchmark bytes: " << measured_bytes << "\n";
  }
  std::cout << "Tile bytes      : " << kStageBytes << " per CTA stage\n";

  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, options.dst_device));

  size_t access_count = measured_bytes / sizeof(uint4);
  size_t tile_count = (access_count + kStageAccesses - 1) / kStageAccesses;
  int blocks = int(std::min<size_t>(std::max<size_t>(1, sm_count * std::max(1, options.blocks_per_sm)), tile_count));
  size_t shared_bytes = size_t(kStages) * kStageBytes;

  CUDA_CHECK(cudaSetDevice(options.src_device));
  uint4* src_buffer = nullptr;
  CUDA_CHECK(cudaMalloc(&src_buffer, measured_bytes));

  int init_blocks = std::max(1, std::min<int>(4096, int((access_count + kThreads - 1) / kThreads)));
  initialize_source_kernel<<<init_blocks, kThreads>>>(src_buffer, access_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaSetDevice(options.dst_device));
  uint4* dst_buffer = nullptr;
  CUDA_CHECK(cudaMalloc(&dst_buffer, measured_bytes));
  CUDA_CHECK(cudaMemset(dst_buffer, 0, measured_bytes));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  std::cout << "Grid/Block      : " << blocks << " x " << kThreads << "\n";
  std::cout << "Shared memory   : " << shared_bytes << " bytes per CTA\n";
  std::cout << "Warmup/iters    : " << options.warmup << " / " << options.iterations << "\n\n";

  BenchmarkResult cp_async_result = benchmark_method(
    "cp.async peer",
    [&](cudaStream_t launch_stream) {
      cp_async_peer_copy_kernel<<<blocks, kThreads, shared_bytes, launch_stream>>>(
        dst_buffer,
        src_buffer,
        access_count);
      CUDA_CHECK(cudaGetLastError());
    },
    stream,
    measured_bytes,
    options.warmup,
    options.iterations,
    options.verify,
    src_buffer,
    dst_buffer,
    access_count,
    options.dst_device);
  print_result(cp_async_result);

  if (options.run_direct) {
    BenchmarkResult direct_result = benchmark_method(
      "direct peer ld/st",
      [&](cudaStream_t launch_stream) {
        direct_peer_copy_kernel<<<blocks, kThreads, 0, launch_stream>>>(
          dst_buffer,
          src_buffer,
          access_count);
        CUDA_CHECK(cudaGetLastError());
      },
      stream,
      measured_bytes,
      options.warmup,
      options.iterations,
      options.verify,
      src_buffer,
      dst_buffer,
      access_count,
      options.dst_device);
    print_result(direct_result);
  }

  if (options.run_memcpy_peer) {
    BenchmarkResult memcpy_result = benchmark_method(
      "cudaMemcpyPeer",
      [&](cudaStream_t launch_stream) {
        CUDA_CHECK(cudaMemcpyPeerAsync(
          dst_buffer,
          options.dst_device,
          src_buffer,
          options.src_device,
          measured_bytes,
          launch_stream));
      },
      stream,
      measured_bytes,
      options.warmup,
      options.iterations,
      options.verify,
      src_buffer,
      dst_buffer,
      access_count,
      options.dst_device);
    print_result(memcpy_result);
  }

  if (options.run_latency) {
    std::cout << "\n";
    LatencyResult latency_result = measure_cp_async_latency(
      src_buffer,
      access_count,
      std::max(1, options.latency_iterations),
      options.latency_stride_bytes,
      options.dst_device);
    print_latency_result(latency_result);
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(dst_buffer));
  CUDA_CHECK(cudaSetDevice(options.src_device));
  CUDA_CHECK(cudaFree(src_buffer));

  return 0;
}
