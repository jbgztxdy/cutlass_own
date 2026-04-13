/***************************************************************************************************
 * K-Slice-Stationary Overlap Benchmark
 *
 * Tests whether K-outer-loop traversal with double-buffered NVLink prefetch can hide
 * communication latency for a multi-GPU GEMM (tile-config=12: TB 128x128x16).
 *
 * Loop structure:
 *   for k_slice = 0 .. K/K_s - 1:
 *     [fetch_stream] 2D async copy A[:, k_slice*K_s:(k+1)*K_s]  from storage GPU
 *     [fetch_stream] 2D async copy B[k_slice*K_s:(k+1)*K_s, :]  from storage GPU
 *     [compute_stream] wait for fetch, then CUTLASS partial GEMM (beta = accumulate)
 *
 * Reports three variants:
 *   [1] Naive remote  : full GEMM, A/B on remote GPU
 *   [2] K-stationary  : K-outer double-buffered prefetch + partial GEMMs
 *   [3] Local bound   : full GEMM, A/B already on compute GPU
 *
 * Usage:
 *   ./k_slice_stationary_benchmark [--storage-device=0] [--compute-device=1]
 *                                   [--m=5120] [--n=4096] [--k=4096]
 *                                   [--k-slice=256]   (multiple of 16)
 *                                   [--iterations=20] [--no-check]
 *
 **************************************************************************************************/

#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/software_cache.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/command_line.h"
#include "helper.h"

///////////////////////////////////////////////////////////////////////////////
// CUTLASS config — tile-config=12: TB(128,128,16) Warp(64,64,16) Inst(16,8,8)
// Using device::Gemm + GemmIdentityThreadblockSwizzle (same as example 14
// single-GPU), which avoids StreamK scheduling overhead and matches the
// ~117 TFLOPs reference performance.
///////////////////////////////////////////////////////////////////////////////

using ElementA     = float;
using ElementB     = float;
using ElementC     = float;
using ElementAccum = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

using ShapeTB   = cutlass::gemm::GemmShape<128, 128, 16>;
using ShapeWarp = cutlass::gemm::GemmShape<64,  64,  16>;
using ShapeMMA  = cutlass::gemm::GemmShape<16,  8,   8>;
constexpr int kNumStages = 4;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
    ElementAccum, float>;

using SwizzleIdentity = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccum,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    ShapeTB, ShapeWarp, ShapeMMA,
    EpilogueOp, SwizzleIdentity, kNumStages>;

///////////////////////////////////////////////////////////////////////////////
// Helpers
///////////////////////////////////////////////////////////////////////////////

static float event_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
    return ms;
}

// Build Arguments for device::Gemm (identity-swizzle variant, TensorRef interface).
static typename Gemm::Arguments make_args(
        int M, int N, int K,
        ElementA const *ptr_A, int lda,
        ElementB const *ptr_B, int ldb,
        ElementC const *ptr_C, ElementC *ptr_D, int ldc,
        float alpha, float beta) {
    typename Gemm::Arguments args{
        {M, N, K},
        {ptr_A, LayoutA(lda)},
        {ptr_B, LayoutB(ldb)},
        {ptr_C, LayoutC(ldc)},
        {ptr_D, LayoutC(ldc)},
        {alpha, beta},
        /*split_k_slices=*/1};
    return args;
}

///////////////////////////////////////////////////////////////////////////////
// K-stationary pipeline: one full pass over K in K_s-thick slices.
// Returns wall-clock ms measured by events on compute_stream.
///////////////////////////////////////////////////////////////////////////////

static float run_k_stationary(
        Gemm &gemm_op,
        int M, int N, int K, int K_s,
        int storage_device, int compute_device,
        ElementA const *d_A_remote, // [M, K] row-major  on storage GPU
        ElementB const *d_B_remote, // [K, N] col-major  on storage GPU
        ElementC *d_D,              // [M, N] output     on compute GPU
        void *d_workspace, size_t ws_bytes,
        cudaStream_t fetch_stream,
        cudaStream_t compute_stream,
        cudaEvent_t *fetch_done,    // [2]
        cudaEvent_t *compute_done)  // [2]
{
    int K_tiles = K / K_s;

    // Double-buffer A/B slice buffers on compute GPU.
    size_t bytes_A_slice = size_t(M) * K_s * sizeof(float);
    size_t bytes_B_slice = size_t(N) * K_s * sizeof(float);

    float *buf_A[2] = {}, *buf_B[2] = {};
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMalloc(&buf_A[i], bytes_A_slice));
        CUDA_CHECK(cudaMalloc(&buf_B[i], bytes_B_slice));
    }

    // Reset compute_done events so first iteration doesn't stall.
    for (int i = 0; i < 2; ++i)
        CUDA_CHECK(cudaEventRecord(compute_done[i], compute_stream));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));

    // ---- Pre-fetch slice 0 into buf[0] ----
    // A-slice[k=0]: M rows x K_s cols from row-major A[M,K] at col offset 0
    //   spitch = K*4 (full row),  dpitch = K_s*4 (slice row), width = K_s*4, height = M
    CUDA_CHECK(cudaMemcpy2DAsync(
        buf_A[0], K_s * sizeof(float),
        d_A_remote, K * sizeof(float),
        K_s * sizeof(float), M,
        cudaMemcpyDefault, fetch_stream));
    // B-slice[k=0]: K_s "rows" x N "cols" from col-major B[K,N] at row offset 0
    //   col-major means column j is at j*K; within a column we want rows 0..K_s-1
    //   spitch = K*4, dpitch = K_s*4, width = K_s*4, height = N
    CUDA_CHECK(cudaMemcpy2DAsync(
        buf_B[0], K_s * sizeof(float),
        d_B_remote, K * sizeof(float),
        K_s * sizeof(float), N,
        cudaMemcpyDefault, fetch_stream));
    CUDA_CHECK(cudaEventRecord(fetch_done[0], fetch_stream));

    // Record start of compute timeline
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start, compute_stream));

    for (int k = 0; k < K_tiles; ++k) {
        int cur  = k & 1;
        int next = cur ^ 1;

        // Kick off next fetch while current compute is running.
        if (k + 1 < K_tiles) {
            // Don't overwrite 'next' buf until compute is done using it
            // (happens when k >= 1, because compute used 'next' at k-1).
            if (k >= 1)
                CUDA_CHECK(cudaStreamWaitEvent(fetch_stream, compute_done[next], 0));

            int col_off = (k + 1) * K_s;
            CUDA_CHECK(cudaMemcpy2DAsync(
                buf_A[next], K_s * sizeof(float),
                d_A_remote + col_off, K * sizeof(float),
                K_s * sizeof(float), M,
                cudaMemcpyDefault, fetch_stream));
            CUDA_CHECK(cudaMemcpy2DAsync(
                buf_B[next], K_s * sizeof(float),
                d_B_remote + col_off, K * sizeof(float),
                K_s * sizeof(float), N,
                cudaMemcpyDefault, fetch_stream));
            CUDA_CHECK(cudaEventRecord(fetch_done[next], fetch_stream));
        }

        // Compute stream waits for current slice.
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, fetch_done[cur], 0));

        // Partial GEMM: D = 1 * A_slice * B_slice + beta * D
        float beta = (k == 0) ? 0.0f : 1.0f;
        auto args = make_args(M, N, K_s,
            buf_A[cur], K_s,
            buf_B[cur], K_s,
            d_D, d_D, N,   // C = D for accumulation
            1.0f, beta);

        cutlass::Status s = gemm_op.initialize(args, d_workspace, compute_stream);
        if (s != cutlass::Status::kSuccess) {
            fprintf(stderr, "initialize failed at k=%d: %s\n",
                    k, cutlass::cutlassGetStatusString(s));
            return -1.f;
        }
        s = gemm_op(compute_stream);
        if (s != cutlass::Status::kSuccess) {
            fprintf(stderr, "run failed at k=%d\n", k);
            return -1.f;
        }

        CUDA_CHECK(cudaEventRecord(compute_done[cur], compute_stream));
    }

    CUDA_CHECK(cudaEventRecord(ev_stop, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = event_ms(ev_start, ev_stop);

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    for (int i = 0; i < 2; ++i) { cudaFree(buf_A[i]); cudaFree(buf_B[i]); }
    return ms;
}

///////////////////////////////////////////////////////////////////////////////
// main
///////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **argv) {
    cutlass::CommandLine cmd(argc, argv);

    if (cmd.check_cmd_line_flag("help")) {
        printf(
            "k_slice_stationary_benchmark\n\n"
            "  --storage-device=<int>  GPU holding A/B  (default 0)\n"
            "  --compute-device=<int>  GPU running GEMM (default 1)\n"
            "  --m=<int>  --n=<int>  --k=<int>          (default 5120/4096/4096)\n"
            "  --k-slice=<int>         K-slice width, multiple of 16 (default 256)\n"
            "  --iterations=<int>      Timed iterations (default 20)\n"
            "  --no-check              Skip correctness check\n");
        return 0;
    }

    int storage_device = 0, compute_device = 1;
    int M = 5120, N = 4096, K = 4096, K_s = 256, iterations = 20;
    bool no_check = cmd.check_cmd_line_flag("no-check");

    cmd.get_cmd_line_argument("storage-device", storage_device);
    cmd.get_cmd_line_argument("compute-device", compute_device);
    cmd.get_cmd_line_argument("m", M);
    cmd.get_cmd_line_argument("n", N);
    cmd.get_cmd_line_argument("k", K);
    cmd.get_cmd_line_argument("k-slice", K_s);
    cmd.get_cmd_line_argument("iterations", iterations);

    if (K % K_s != 0 || K_s % 16 != 0) {
        fprintf(stderr, "k-slice=%d must divide K=%d and be a multiple of 16\n", K_s, K);
        return 1;
    }

    double gflops = 2.0 * M * N * K / 1e9;

    printf("=================================================================\n");
    printf("  K-Slice-Stationary Benchmark  (tile-config=12)\n");
    printf("  M=%-5d N=%-5d K=%-5d  K_slice=%d (%d iterations)\n",
           M, N, K, K_s, K / K_s);
    printf("  Storage GPU: %d   Compute GPU: %d\n", storage_device, compute_device);
    printf("  Timed iterations: %d\n", iterations);
    printf("=================================================================\n\n");

    // ---- P2P setup ----
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) { fprintf(stderr, "Need >= 2 GPUs\n"); return 1; }

    CUDA_CHECK(cudaSetDevice(compute_device));
    if (compute_device != storage_device) {
        int ok = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&ok, compute_device, storage_device));
        if (!ok) {
            fprintf(stderr, "GPU%d cannot P2P-access GPU%d\n",
                    compute_device, storage_device);
            return 1;
        }
        cudaError_t e = cudaDeviceEnablePeerAccess(storage_device, 0);
        if (e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
    }

    // ---- Allocate A/B on storage GPU ----
    CUDA_CHECK(cudaSetDevice(storage_device));
    size_t bytes_A = size_t(M) * K * sizeof(float);
    size_t bytes_B = size_t(K) * N * sizeof(float);
    float *d_A_remote = nullptr, *d_B_remote = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_remote, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B_remote, bytes_B));

    // Host init for correctness check
    std::vector<float> h_A(size_t(M) * K), h_B(size_t(K) * N);
    srand(42);
    for (auto &v : h_A) v = float(rand() % 17 - 8) * 0.125f;
    for (auto &v : h_B) v = float(rand() % 17 - 8) * 0.125f;
    CUDA_CHECK(cudaMemcpy(d_A_remote, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_remote, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    // ---- Allocate output + local copies on compute GPU ----
    CUDA_CHECK(cudaSetDevice(compute_device));
    size_t bytes_D = size_t(M) * N * sizeof(float);

    float *d_D = nullptr, *d_C_zero = nullptr;
    CUDA_CHECK(cudaMalloc(&d_D, bytes_D));
    CUDA_CHECK(cudaMalloc(&d_C_zero, bytes_D));
    CUDA_CHECK(cudaMemset(d_C_zero, 0, bytes_D));

    float *d_A_local = nullptr, *d_B_local = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A_local, bytes_A));
    CUDA_CHECK(cudaMalloc(&d_B_local, bytes_B));
    CUDA_CHECK(cudaMemcpyPeer(d_A_local, compute_device, d_A_remote, storage_device, bytes_A));
    CUDA_CHECK(cudaMemcpyPeer(d_B_local, compute_device, d_B_remote, storage_device, bytes_B));

    // ---- Pre-allocate CUTLASS workspace ----
    // device::Gemm with split_k=1 and identity swizzle needs no reduction workspace;
    // get_workspace_size() typically returns 0.  Allocate max of full and partial.
    Gemm gemm_op;
    auto probe_full    = make_args(M, N, K,   d_A_local, K,   d_B_local, K,   d_C_zero, d_D, N, 1.f, 0.f);
    auto probe_partial = make_args(M, N, K_s, d_A_local, K_s, d_B_local, K_s, d_C_zero, d_D, N, 1.f, 0.f);
    size_t ws_bytes = std::max(Gemm::get_workspace_size(probe_full),
                               Gemm::get_workspace_size(probe_partial));
    void *d_workspace = nullptr;
    if (ws_bytes) CUDA_CHECK(cudaMalloc(&d_workspace, ws_bytes));

    // ---- Streams and events ----
    cudaStream_t fetch_stream, compute_stream;
    CUDA_CHECK(cudaStreamCreate(&fetch_stream));
    CUDA_CHECK(cudaStreamCreate(&compute_stream));

    cudaEvent_t fetch_done[2], compute_done[2];
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreate(&fetch_done[i]));
        CUDA_CHECK(cudaEventCreate(&compute_done[i]));
    }

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    // ===========================================================================
    // [1] Naive remote: full GEMM, A/B on storage GPU
    // ===========================================================================
    printf("--- [1] Naive remote (A/B on GPU%d, GEMM on GPU%d) ---\n",
           storage_device, compute_device);
    {
        auto args = make_args(M, N, K, d_A_remote, K, d_B_remote, K, d_C_zero, d_D, N, 1.f, 0.f);
        CUTLASS_CHECK(gemm_op.can_implement(args));
        CUTLASS_CHECK(gemm_op.initialize(args, d_workspace, compute_stream));

        // Warmup
        for (int i = 0; i < 3; ++i) gemm_op(compute_stream);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        CUDA_CHECK(cudaEventRecord(ev0, compute_stream));
        for (int i = 0; i < iterations; ++i) gemm_op(compute_stream);
        CUDA_CHECK(cudaEventRecord(ev1, compute_stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));
    }
    float naive_ms = event_ms(ev0, ev1) / iterations;
    printf("  Avg: %.3f ms   TFLOPs: %.2f\n\n", naive_ms, gflops / naive_ms);

    // Save naive result for comparison
    std::vector<float> h_D_naive(size_t(M) * N);
    CUDA_CHECK(cudaMemcpy(h_D_naive.data(), d_D, bytes_D, cudaMemcpyDeviceToHost));

    // ===========================================================================
    // [3] Local upper bound
    // ===========================================================================
    printf("--- [3] Local upper bound (A/B on GPU%d) ---\n", compute_device);
    {
        auto args = make_args(M, N, K, d_A_local, K, d_B_local, K, d_C_zero, d_D, N, 1.f, 0.f);
        CUTLASS_CHECK(gemm_op.initialize(args, d_workspace, compute_stream));

        for (int i = 0; i < 3; ++i) gemm_op(compute_stream);
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));

        CUDA_CHECK(cudaEventRecord(ev0, compute_stream));
        for (int i = 0; i < iterations; ++i) gemm_op(compute_stream);
        CUDA_CHECK(cudaEventRecord(ev1, compute_stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));
    }
    float local_ms = event_ms(ev0, ev1) / iterations;
    printf("  Avg: %.3f ms   TFLOPs: %.2f\n\n", local_ms, gflops / local_ms);

    // ===========================================================================
    // [3b] Local K-slice baseline: same loop structure as K-stationary,
    //      but fetch copies from d_A_local / d_B_local (local HBM, no NVLink).
    //      This is the true upper bound for K-stationary: it isolates the
    //      overhead of the K-slice loop structure (multiple partial GEMMs,
    //      beta=1 D re-reads, initialize() calls) from NVLink latency.
    // ===========================================================================
    printf("--- [3b] Local K-slice baseline (same loop, fetch from local HBM) ---\n");
    {
        // Run k_stationary but with storage==compute so cudaMemcpy2DAsync
        // copies within the same GPU (local HBM → local HBM).
        // We achieve this by passing d_A_local / d_B_local as the "remote" source
        // on the same device.  To keep the copy-engine path identical, we just
        // run run_k_stationary with storage_device == compute_device and
        // the local pointers.
        for (int i = 0; i < 2; ++i) {  // warmup
            CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));
            run_k_stationary(gemm_op, M, N, K, K_s, compute_device, compute_device,
                             d_A_local, d_B_local, d_D,
                             d_workspace, ws_bytes,
                             fetch_stream, compute_stream,
                             fetch_done, compute_done);
        }
        float total_lks = 0.f;
        for (int iter = 0; iter < iterations; ++iter) {
            CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));
            float ms = run_k_stationary(gemm_op, M, N, K, K_s, compute_device, compute_device,
                                        d_A_local, d_B_local, d_D,
                                        d_workspace, ws_bytes,
                                        fetch_stream, compute_stream,
                                        fetch_done, compute_done);
            total_lks += ms;
        }
        float lks_ms = total_lks / iterations;
        printf("  Avg: %.3f ms   TFLOPs: %.2f\n\n", lks_ms, gflops / lks_ms);
    }

    // ===========================================================================
    // [2] K-stationary: K-outer loop with double-buffered NVLink prefetch
    // ===========================================================================
    printf("--- [2] K-stationary (K_s=%d, %d K-slices, double-buffered) ---\n",
           K_s, K / K_s);

    // Warmup
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));
        run_k_stationary(gemm_op, M, N, K, K_s, storage_device, compute_device,
                         d_A_remote, d_B_remote, d_D,
                         d_workspace, ws_bytes,
                         fetch_stream, compute_stream,
                         fetch_done, compute_done);
    }

    float total_kstat_ms = 0.f;
    for (int iter = 0; iter < iterations; ++iter) {
        CUDA_CHECK(cudaMemset(d_D, 0, bytes_D));
        float ms = run_k_stationary(gemm_op, M, N, K, K_s, storage_device, compute_device,
                                    d_A_remote, d_B_remote, d_D,
                                    d_workspace, ws_bytes,
                                    fetch_stream, compute_stream,
                                    fetch_done, compute_done);
        if (ms < 0) return 1;
        total_kstat_ms += ms;
    }
    float kstat_ms = total_kstat_ms / iterations;
    printf("  Avg: %.3f ms   TFLOPs: %.2f\n", kstat_ms, gflops / kstat_ms);

    // Correctness check
    if (!no_check) {
        std::vector<float> h_D_kstat(size_t(M) * N);
        CUDA_CHECK(cudaMemcpy(h_D_kstat.data(), d_D, bytes_D, cudaMemcpyDeviceToHost));
        double max_err = 0, ref_max = 0;
        for (size_t i = 0; i < h_D_naive.size(); ++i) {
            double diff = std::fabs(double(h_D_naive[i]) - double(h_D_kstat[i]));
            max_err = std::max(max_err, diff);
            ref_max = std::max(ref_max, std::fabs(double(h_D_naive[i])));
        }
        double rel = ref_max > 0 ? max_err / ref_max : max_err;
        printf("  Correctness vs naive: max_abs=%.3e  rel=%.3e  %s\n",
               max_err, rel, rel < 1e-3 ? "PASSED" : "FAILED");
    }
    printf("\n");

    // ===========================================================================
    // Summary
    // ===========================================================================
    double transfer_total_GB = double(bytes_A + bytes_B) / 1e9;
    double kstat_comm_fraction =
        (naive_ms - local_ms) > 0.001 ?
        (kstat_ms - local_ms) / (naive_ms - local_ms) : 0.0;

    printf("=================================================================\n");
    printf("  SUMMARY: M=%d N=%d K=%d  K_slice=%d\n", M, N, K, K_s);
    printf("  Total data from remote: %.1f GB\n", transfer_total_GB);
    printf("-----------------------------------------------------------------\n");
    printf("  %-18s | %8s | %6s | %14s\n",
           "Variant", "ms/iter", "TFLOPs", "Speedup vs naive");
    printf("  %-18s | %8.3f | %6.2f | %14.2fx\n",
           "Local (upper bound)", local_ms, gflops/local_ms, naive_ms/local_ms);
    printf("  %-18s | %8.3f | %6.2f | %14.2fx\n",
           "K-stationary", kstat_ms, gflops/kstat_ms, naive_ms/kstat_ms);
    printf("  %-18s | %8.3f | %6.2f | %14s\n",
           "Naive remote", naive_ms, gflops/naive_ms, "baseline");
    printf("-----------------------------------------------------------------\n");
    printf("  K-stat gap to ideal  : %.2fx  (%.1f%% of ideal perf)\n",
           kstat_ms / local_ms, 100.0 * local_ms / kstat_ms);
    printf("  NVLink overhead hidden: %.1f%%  (0%% = no help, 100%% = fully hidden)\n",
           100.0 * (1.0 - kstat_comm_fraction));
    printf("=================================================================\n");

    // ---- Cleanup ----
    CUDA_CHECK(cudaSetDevice(storage_device));
    cudaFree(d_A_remote);
    cudaFree(d_B_remote);

    CUDA_CHECK(cudaSetDevice(compute_device));
    cudaFree(d_D);
    cudaFree(d_C_zero);
    cudaFree(d_A_local);
    cudaFree(d_B_local);
    if (d_workspace) cudaFree(d_workspace);
    cudaStreamDestroy(fetch_stream);
    cudaStreamDestroy(compute_stream);
    for (int i = 0; i < 2; ++i) {
        cudaEventDestroy(fetch_done[i]);
        cudaEventDestroy(compute_done[i]);
    }
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    return 0;
}
