# CUTLASS Example 14 迭代器与计算顺序分析报告

## 命令行入口与模板实例化

运行命令：
```
./examples/14_ampere_tf32_tensorop_gemm_multigpu_new/14_ampere_tf32_tensorop_gemm_multigpu_new \
    --tile-config=12 --storage-device=0 --compute-device=1 --software-cache=1
```

**文件**：`examples/14_ampere_tf32_tensorop_gemm_multigpu_new/ampere_tf32_tensorop_gemm.cu`

该例子以宏 `TILE_CONFIG_LIST` 展开 27 种 tile 配置（ID 0–26）。`--tile-config=12` 对应：

```cpp
// ID 12: TB(128,128,16), Warp(64,64,16), Inst(16,8,8)
using GemmCfg12 = GemmKernel<ShapeTB_128_128_16, ShapeWarp_64_64_16>;
```

顶层类型链：
```
cutlass::gemm::device::GemmUniversal
  └─ cutlass::gemm::kernel::GemmUniversalStreamk          (gemm_universal_streamk.h)
       └─ cutlass::gemm::threadblock::MmaMultistage        (mma_multistage.h)
            ├─ IteratorA = PredicatedTileIterator<...>     (predicated_tile_iterator.h)
            ├─ IteratorB = PredicatedTileIterator<...>
            └─ 基类 MmaBase                                (mma_base.h)
                 ├─ warp_tile_iterator_A_ (从 smem 读 A)
                 └─ warp_tile_iterator_B_ (从 smem 读 B)
```

---

## 关键参数（以 tile-config=12 为例）

| 参数 | 值 |
|------|----|
| ThreadBlock tile (M,N,K) | 128, 128, 16 |
| Warp tile (M,N,K) | 64, 64, 16 |
| MMA 指令 (M,N,K) | 16, 8, 8 |
| NumStages | 4 |
| WarpCount (M,N,K) | 128/64=2, 128/64=2, 16/16=1 → 4 个 warp |
| kWarpGemmIterations | WarpK / MmaK = 16 / 8 = 2 |
| 每个 threadblock 线程数 | 4 warps × 32 = 128 |
| Swizzle | `ThreadblockSwizzleStreamK` |

---

## 整体调度层次（从粗到细）

### 第一层：设备内核 `gemm()` — Block 角色分配

**文件**：`include/cutlass/gemm/kernel/gemm_universal_streamk.h`，`GemmUniversalStreamk::gemm()`（行 1018–1126）

每个 block 根据 `block_idx` 被分配为三类角色之一：

```
block_idx:
┌──────────────────────────────┬───────────────────┬────────────────────┐
│  SK 块 (sk_block)            │  DP 块 (dp_block) │  Reduce 块         │
│  [0, sk_padding_start)       │  [dp_start, red)  │  [red, grid_pad)   │
└──────────────────────────────┴───────────────────┴────────────────────┘
```

- **DP 块**（Data-Parallel）：每个 block 拥有完整 K 范围的若干 output tile，`k_begin=0, k_end=K`。
- **SK 块**（Stream-K）：每个 block 拥有若干连续的全局迭代 `[block_iter_begin, block_iter_end)`，可能跨越多个 output tile 的部分 K 区间。
- **Reduce 块**：仅在 `kMixed` 策略下出现，负责聚合 SK 块写入全局 workspace 的 partial accumulators。

### 第二层：Output tile 在 C 矩阵中的排列顺序

**文件**：`include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h`，`get_tile_offset()`（行 661–690）

tile_idx 到 (m, n) 坐标的映射逻辑：
- **默认**：row-major 光栅，即先沿 N 方向增加，再沿 M 方向增加：
  ```
  m = tile_idx / tiled_shape_n
  n = tile_idx % tiled_shape_n
  ```
- **column-major 光栅**：当 `tiled_shape.m() < tiled_shape.n()` 时改用列主序：
  ```
  n = tile_idx / tiled_shape_m
  m = tile_idx % tiled_shape_m
  ```
- **cohort raster**：以 8×4 CTA 的 cohort 为单位做局部分块排列（提升 L2 命中）。

**DP 块的 tile 遍历顺序**（持久化内核模式，行 1112–1116）：
```cpp
tile_idx += params.block_mapping.avail_sms;  // 每次跨一个 wave 宽度
```
block 0 处理 tile 0, `avail_sms`, `2*avail_sms`, ...；block 1 处理 tile 1, `1+avail_sms`, ...

**SK 块的 tile 遍历顺序**（行 1119–1122）：
```cpp
tile_idx--;  // 从 block 覆盖范围的最高 tile_idx 开始，反向遍历
```

### 第三层：每个 tile 内 K 方向的迭代——`process_tile` 与 `MmaMultistage::operator()`

**文件**：`gemm_universal_streamk.h`，`process_tile()`（行 946–1013）

```cpp
// 初始化全局内存迭代器，从 (m_begin, k_begin) 出发
IteratorA iterator_A = init_iterator_A(tile_work, ...);
// init_iterator_A 关键部分（行 670–677）：
//   m_begin = tile_work.tiled_coord.m() * Mma::Shape::kM
//   构造时传入偏移 {m_begin, tile_work.k_begin}
// 对于 SK 块，k_begin 可能是 K 的中间某个值（而非 0）

mma(tile_work.k_iters_remaining, accum, iterator_A, iterator_B, accum);
```

进入 `MmaMultistage::operator()`（`mma_multistage.h` 行 1004–1041）：

```cpp
void operator()(int gemm_k_iterations, ...) {
    // 1. prologue：预取前 kStages-1=3 个 K-tile 到 smem
    prologue(iterator_A, iterator_B, gemm_k_iterations);

    // 2. 等待第 0 个 stage 的 cp.async 完成
    gmem_wait(...);

    // 3. 初始化 accum
    accum = src_accum;

    // 4. 主循环
    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B, ...);
}
```

---

## `MmaMultistage` 内部 K 迭代详解

### Prologue（行 585–705）

预取前 `kStages - 1`（= 3）个 K-tile 段：

```cpp
for (int stage = 0; stage < kStages - 1; ++stage, --gemm_k_iterations) {
    // 对每个 stage，每个 thread 发出 AsyncCopyIterationsPerStageA 条 cp.async
    for (int j = 0; j < AsyncCopyIterationsPerStageA; ++j) {
        cp_async_zfill(smem_iterator_A_.get(), iterator_A.get(), ...);
        ++iterator_A;       // 在当前 K-tile 内向前走一步
        ++smem_iterator_A_;
    }
    // B 同理...
    advance_smem_write_stage(iterator_A, iterator_B); // 推进到下一 K-tile
    cp_async_fence();
}
```

`advance_smem_write_stage()`（行 365–386）做两件事：
1. 全局迭代器跳到下一个 K-tile：`iterator_A.add_tile_offset({0, 1})`（A 的 K 维是 contiguous 的列方向）
2. smem 写迭代器也跟进，并在超出 `kStages` 时绕回环形缓冲区头部

### 主循环 `gemm_iters`（行 900–964）

```cpp
// 先 load 第 0 个 kgroup 的 warp fragment
warp_tile_iterator_A_.set_kgroup_index(0);
warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);

// 主循环：gemm_k_iterations 计数在每个 outer 迭代的 mac_loop_iter 内递减
for (; gemm_k_iterations > (-kStages + 1);) {
    mac_loop_iter(..., gemm_k_iterations);
}
```

### 最内层循环 `mac_loop_iter`（行 764–894）

每次调用展开 `kWarpGemmIterations`（= 2，对于 tile-config=12）次 warp-level MMA。  
以下以 `warp_mma_k = 0, 1` 描述（对于 kStages=4, kWarpGemmIterations=2 的情况）：

```
warp_mma_k = 0:
  (a) load 下一个 warp fragment: set_kgroup_index(1), load frag_A[1], ++warp_tile_iterator_A_
  (b) warp_mma_k > 0 才 transform，所以此处跳过
  (c) 执行 MMA: warp_mma_(accum, transformed_frag_A[0], transformed_frag_B[0], accum)
  (d) warp_mma_k < kWarpGemmIterations-1=1，发出第 0 组 cp.async:
      copy_tiles_and_advance(iterator_A, iterator_B, group_start=0)

warp_mma_k = 1:
  (a) load 下一个 warp fragment: set_kgroup_index(0 % 2=0), load frag_A[0], ++warp_tile_iterator_A_
  (b) warp_mma_k > 0: transform(frag_A[1], ...)   ← 为本次 MMA 准备
  (c) 执行 MMA: warp_mma_(accum, transformed_frag_A[1], transformed_frag_B[1], accum)
  (d) warp_mma_k == kWarpGemmIterations-2=0? 不等，跳过
  (e) warp_mma_k+2 == kWarpGemmIterations=2: 真（1+2 ≠ 2，但 0+2==2 才对...）
      → 实际上此分支在 warp_mma_k = kWarpGemmIterations-2 时触发
      → 对于 kWarpGemmIterations=2：warp_mma_k=0 触发 "second-to-last" 分支
      → 发出最后一组 cp.async，然后：
          cp_async_fence()
          gmem_wait()      ← 等待最老的 stage 提交
          advance_smem_write_stage()   ← 全局迭代器 +1 K-tile
          advance_smem_read_stage()    ← smem 读指针移到下一 stage
          --gemm_k_iterations
  (f) warp_mma_k+1 == kWarpGemmIterations=2: 真（1+1=2）
      → transform 下一次迭代开头的 fragment[0]

外层循环回到 warp_mma_k=0，但此时 frag_A[0] 已是新 stage 的数据
```

**关键：全局 K-tile 的推进点**

在 `mac_loop_iter` 的 "second-to-last" 分支（`warp_mma_k + 2 == kWarpGemmIterations`）中：
```cpp
advance_smem_write_stage(iterator_A, iterator_B);
```
这才是全局迭代器 `iterator_A` / `iterator_B` 向前移动一个 K-tile 的时刻。  
从线性流程看，每经过完整的一次 `mac_loop_iter`（含所有 warp_mma_k 子迭代），全局 K 推进一步。

---

## `PredicatedTileIterator`：全局内存访问细节

**文件**：`include/cutlass/transform/threadblock/predicated_tile_iterator.h`

### 构造时的 thread 分配

```cpp
IteratorA(params, ptr_A, {m_end, k_end}, threadIdx.x, {m_begin, k_begin})
```

构造时，通过 `ThreadMap` 将 `threadIdx.x` 映射到 tile 内的 (row_offset, col_offset)。  
对于 A（RowMajor），ThreadMap 将 128 个线程分配到 128(M) × 16(K) 的 tile：
- 每个线程负责若干 `(strided, contiguous)` 位置的访问
- 具体映射由 `ThreadMap::Iterations::{kStrided, kContiguous}` 决定

### `++iterator_A`（行 281–288）

```cpp
PredicatedTileIterator &operator++() {
    if (kAdvanceRank)
        address_iterator_.add_tile_offset({0, 1});  // 沿 strided 方向推进 1 tile
    else
        address_iterator_.add_tile_offset({1, 0});  // 沿 contiguous 方向推进 1 tile
    return *this;
}
```

对于 A（RowMajor，`AdvanceRank=1`，沿 K 方向推进），`++` 使地址跳到 K+kK 位置。

### `add_tile_offset({0, 1})`（在 `advance_smem_write_stage` 中调用）

与 `++` 效果相同，将 A 的全局指针推进 `kK * sizeof(float)` 字节（即一个 K-tile 的宽度）。

### 每个 thread 的 cp.async 访问模式（`copy_tiles_and_advance`，行 503–580）

```cpp
// 每次调用，每个 thread 发出 kAccessesPerGroupA 条 cp.async：
for (int j = 0; j < kAccessesPerGroupA; ++j) {
    if (group_start_A + j < AsyncCopyIterationsPerStageA) {
        dst_ptr = smem_iterator_A_.get();
        for (int v = 0; v < kAccessesPerVector; ++v) {
            cp_async_zfill<kSrcBytes>(dst_ptr + v, iterator_A.get(), valid);
            ++iterator_A;   // 在 tile 内移到下一个访问位置
        }
        ++smem_iterator_A_;
    }
}
```

`AsyncCopyIterationsPerStageA = IteratorA::ThreadMap::Iterations::kCount`，即每个 thread 为一个 stage 发出的总 cp.async 数量。这些访问分 `kWarpGemmIterations` 组交错穿插在 MMA 计算之间，以隐藏全局内存延迟。

---

## smem 的环形缓冲区结构

**文件**：`mma_base.h`，`SharedStorage`（行 140–196）

smem 中 A 的布局（矩阵形状）：
```
ShapeA = (kM + padding_row) × (kK * kStages + padding_col)
       = (128 + 0) × (16 * 4 + padding) [tile-config=12]
```

4 个 stage 在 K 维方向上线性排列，写指针 `smem_write_stage_idx_` 在 0..3 之间循环：
```
[stage 0 | stage 1 | stage 2 | stage 3]
 ↑ 写入新数据（cp.async 目标）
                              ↑ MMA 读取（最旧的 committed stage）
```

**写指针推进**（`advance_smem_write_stage`，行 365–386）：
```cpp
smem_iterator_A_.add_tile_offset({0, 1});  // smem 内 K 方向 +1 tile
++smem_write_stage_idx_;
if (smem_write_stage_idx_ == kStages) {
    smem_iterator_A_.add_tile_offset({0, -kStages});  // 回绕到开头
    smem_write_stage_idx_ = 0;
}
```

**读指针推进**（`advance_smem_read_stage`，行 351–360）：
```cpp
++smem_read_stage_idx_;
if (smem_read_stage_idx_ == kStages) {
    warp_tile_iterator_A_.add_tile_offset({0,
        -kStages * kPartitionsK * kWarpGemmIterations});
    smem_read_stage_idx_ = 0;
}
```

---

## Warp 的分工与 smem 读取位置

**文件**：`mma_multistage.h`，构造函数（行 336–347）

```cpp
int warp_idx_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
int warp_idx_m = warp_idx_mn % WarpCount::kM;   // 0 或 1
int warp_idx_n = warp_idx_mn / WarpCount::kM;   // 0 或 1

// 每个 warp 的 smem 读起始位置偏移到其负责的 M/N 分区
warp_tile_iterator_A_.add_tile_offset({warp_idx_m, kWarpGemmIterations * warp_idx_k});
warp_tile_iterator_B_.add_tile_offset({kWarpGemmIterations * warp_idx_k, warp_idx_n});
```

对于 tile-config=12（WarpCount=2×2×1）：
| warp_idx | (warp_idx_m, warp_idx_n) | A smem 读取的 M 行范围 | B smem 读取的 N 列范围 |
|----------|--------------------------|----------------------|----------------------|
| 0        | (0, 0)                   | 行 [0, 64)           | 列 [0, 64)           |
| 1        | (1, 0)                   | 行 [64, 128)         | 列 [0, 64)           |
| 2        | (0, 1)                   | 行 [0, 64)           | 列 [64, 128)         |
| 3        | (1, 1)                   | 行 [64, 128)         | 列 [64, 128)         |

每个 warp 独立完成其 64×64×kK 的 warp-level GEMM，并将结果累加到私有的 `accum` fragment。

### Warp 内的 kgroup 迭代

在 `mac_loop_iter` 内，`set_kgroup_index((warp_mma_k + 1) % kWarpGemmIterations)` 选择 smem 中 K 方向的第几个分组（即第几个 8 元素宽的 K-slice）。对于 `kWarpGemmIterations=2`：
- `warp_mma_k=0` 时，先读 kgroup 1，执行 kgroup 0 的 MMA
- `warp_mma_k=1` 时，先读 kgroup 0（下一 stage），执行 kgroup 1 的 MMA

---

## StreamK 中 K 分割 tile 的迭代顺序

**文件**：`gemm_universal_streamk.h`，`init_sk_tile_work()`（行 736–770）和主循环（行 1091–1124）

### SK 块的 iteration 分配

`get_iter_extents()` 均匀地将 `sk_iters`（= `sk_tiles × iters_per_tile`）分配给 `sk_blocks` 个 block：
```
block 0: iters [0,   N)
block 1: iters [N,   2N)
...
block k: iters [kN, (k+1)N)
```
每个 block 的 `k_begin = k_iter_begin * kK`，`k_end = k_iter_end * kK`。

### SK 块内部：反向 tile 遍历

SK 块从其迭代范围的**最高 tile** 开始，逆序处理（行 1119–1122）：
```cpp
tile_idx--;  // 每次处理完一个 tile，tile_idx 减一
init_sk_tile_work(tile_work, tile_idx, block_iter_begin, block_iter_begin + block_iters_remaining);
```

对于跨越多个 tile 的 SK 块（如 block 覆盖了 tile 2 的后半段 + tile 1 的全部 + tile 0 的前半段），处理顺序为：
1. 先处理 tile 2 的 `[k_mid, K)` 部分
2. 再处理 tile 1 的 `[0, K)` 部分
3. 最后处理 tile 0 的 `[0, k_end]` 部分

每处理完一段，若是"非 finishing"块（即未覆盖 tile 的 K=0 起始点），则将 partial accumulator 写入全局 workspace；若是"finishing"块（覆盖了 K=0），则从 workspace 聚合 peer partials 后写出 C。

### K-tile 在单个 tile 内的迭代顺序

单个 tile 内，K 从 `k_begin` 到 `k_end` **严格递增**，步长为 `kK`（=16 for tile-config=12）：
- prologue 预取 [k_begin, k_begin + 3*kK) 的 3 个 stage
- 主循环每 `mac_loop_iter` 消耗一个 K-tile，`iterator_A.add_tile_offset({0, 1})` 推进
- 顺序：`k_begin`, `k_begin+kK`, `k_begin+2*kK`, ..., `k_end-kK`

---

## 单个 Thread 的迭代视角

以 `threadIdx.x = 0`（warp 0，lane 0）为例，tile-config=12，问题大小 M=5120, N=4096, K=4096：

### 1. 全局内存迭代器（`PredicatedTileIterator`）

线程 0 在 ThreadMap 中被映射到 tile 内某固定的 (row_offset, col_offset) 集合。每个 thread 在一个 K-stage（16 个 K 元素）内负责加载若干行的数据。

在每次 `copy_tiles_and_advance` 中，线程 0 发出若干 `cp.async` 指令，将全局内存中从 `(m_begin + row_offset, k_begin + col_offset)` 开始的若干 float 值复制到 smem 的对应位置。

内层的 `++iterator_A` 在 tile 内 K 方向或 M 方向步进（具体由 ThreadMap 的 iteration 顺序决定：先 contiguous 后 strided，即先沿 K 步进，再换行）。

### 2. smem 写迭代器（`SmemIteratorA`）

与全局迭代器同步步进，将 cp.async 的目标地址写入 smem 的当前 stage 分区。

### 3. warp-level MMA fragment 加载（`warp_tile_iterator_A_`）

warp 0 的 `warp_tile_iterator_A_` 固定在 smem 中 M 行 [0, 64) 的区域。

每次 `set_kgroup_index(k)` + `load(frag)` + `++`，加载该 warp 在当前 stage、第 k 个 kgroup 内的 A fragment（对应 K 方向上 8 个元素的 warp 分量）。

lane 0（线程 0）在 warp 中负责加载若干特定寄存器位置的数据，具体由 `ldmatrix` 指令的线程布局决定。

### 4. Tensor Core MMA 执行（`warp_mma_`）

调用 `mma.sync` 指令，lane 0 对其持有的 fragment 做 16×8×8 的矩阵乘累加，结果写入 `accum` 寄存器（每个 thread 持有 accum 的一个分片）。

### 5. 完整的时间线（以一个 tile, K=4096 为例，kK=16）

```
K 迭代数 = 4096 / 16 = 256

prologue:  发出 stage 0, 1, 2 的 cp.async（3 × AsyncCopyIterationsPerStageA 条）
gmem_wait: 等待 stage 0 提交

gemm_iters 主循环（256 次外层迭代，每次对应 1 个 K-tile）：

  外层迭代 0（消耗 K = [0, 16)，来自 stage 0）：
    warp_mma_k=0: load kgroup 1 from smem, MMA on kgroup 0, issue cp.async group 0
    warp_mma_k=1: load kgroup 0(next stage), MMA on kgroup 1,
                  issue cp.async group 1, fence, wait, advance stage, --k_iters
                  transform next frag

  外层迭代 1（消耗 K = [16, 32)，来自 stage 1）：
    ...

  外层迭代 255（消耗 K = [4080, 4096)，来自最后一个 stage）：
    ...（clear_mask 使 cp.async 变为 NO-OP）

epilogue: cp_async_wait<0>, __syncthreads, 写出 C
```

---

## 补充：各类 Block 的数量、编号与详细执行方式

### 运行环境假设

以下以 **NVIDIA A100（108 个 SM，sm_occupancy=1）** 和默认问题规模 **M=5120, N=4096, K=4096**，`--tile-config=12`（TB 128×128×16）为例做完整数值推导。

---

### 第一步：确定 tile 的数量

```
tiled_shape_m = ceil(5120 / 128) = 40
tiled_shape_n = ceil(4096 / 128) = 32
output_tiles  = 40 × 32 = 1280
iters_per_tile = ceil(4096 / 16) = 256   (每个 K-tile 16 个元素，沿 K 方向共 256 步)
```

---

### 第二步：调用 `get_blocks()` 决定 SK/DP 比例

```
full_waves       = 1280 / 108 = 11（余 92）
full_wave_tiles  = 11 × 108  = 1188
partial_wave_tiles = 1280 - 1188 = 92
```

由于 `full_waves(11) >= sm_occupancy(1)`，且 `sm_occupancy == 1` 所以第二个 `if` 也不成立，
走"合并最后一个完整 wave 与 partial wave"路径：

```
dp_tiles = full_wave_tiles - avail_sms = 1188 - 108 = 1080
sk_tiles = output_tiles - dp_tiles   = 1280 - 1080 = 200

调用 get_sk_blocks(sk_tiles=200, iters_per_tile=256, avail_sms=108,
                   max_sk_occupancy=1, allow_partial_wave=false)

  sk_iters = 200 × 256 = 51200
  min_sk_blocks = max_sk_blocks = 108  (allow_partial_wave=false 且 max_occ=1)
  → 只有 trial_sk_blocks = 108 一种选项

  sk_iters_per_normal_block = 51200 / 108 = 474
  extra_sk_iters = 51200 - 474×108 = 8
  → 8 个 "big blocks"（每个 475 次迭代），100 个 "normal blocks"（每个 474 次迭代）
  → sk_blocks = 108
```

---

### 第三步：检查是否启用 cohort raster

```
tiled_cohort_shape_m = ceil(40/8) = 5
tiled_cohort_shape_n = ceil(32/4) = 8
cohort_blocks = 5 × 8 × 32 = 1280

cohort_efficiency = dp_blocks(1080) / cohort_blocks(1280) = 0.844 < 0.85

→ 不启用 cohort raster
```

---

### 第四步：waveset 调整（sk_waves=1, sm_occupancy=1）

```
dp_tile_waves    = ceil(1080 / 108) = 10
full_dp_tile_waves = 1080 / 108 = 10
waveset_excess   = (sk_waves + dp_tile_waves) % sm_occupancy = (1+10) % 1 = 0

→ dp_first_wave_tiles 不变，仍为 1
→ dp_blocks = dp_tiles = 1080
```

---

### 第五步：检查是否有 reduction blocks

```
条件：kReductionStrategy == kMixed(✓) AND sk_waves(1) < sm_occupancy(1)(✗)
→ reduction_blocks = 0
```

---

### 最终 Block 分配表

| 类型 | 数量 | block_idx 编号范围 | 备注 |
|------|------|-------------------|------|
| SK 块 | 108 | **[0, 107]** | 其中 0–7 为 big blocks (475 iters)，8–107 为 normal blocks (474 iters) |
| DP 块 | 1080 | **[108, 1187]** | dp_first_wave_tiles=1，每块每次处理 1 个完整 tile |
| Reduce 块 | 0 | —— | 不启用 |
| 总 grid 大小 | **1188** | blockDim.x = 128 | |

索引边界变量（`gemm()` 函数内）：
```
sk_padding_start_block_idx = 1 × 108 = 108     (sk_regions × sk_blocks_per_region)
dp_start_block_idx         = 108               (sk_waves × avail_sms)
reduce_start_block_idx     = 108 + 1080 = 1188
grid_padding_start_block_idx = 1188 + 0 = 1188
```

因此：
- `block_idx ∈ [0, 107]` → `sk_block = true`
- `block_idx ∈ [108, 1187]` → `dp_block = true`

---

### SK Block 的详细执行过程

#### 迭代分配

`get_iter_extents(block_idx, begin, end)` 将 51200 次迭代分配如下：

```
block 0  (big):   iters [0,    475)
block 1  (big):   iters [475,  950)
...
block 7  (big):   iters [3325, 3800)
block 8  (normal): iters [3800, 4274)
block 9  (normal): iters [4274, 4748)
...
block 107(normal): iters [50726, 51200)
```

每个"迭代"对应一个 K-tile（16 个 K 元素）。
- 迭代编号 `iter ∈ [0, 256)` 属于 tile 0（output tile 索引 0）
- 迭代编号 `iter ∈ [256, 512)` 属于 tile 1
- 迭代编号 `iter ∈ [256*i, 256*(i+1))` 属于 tile i
- 总共 200 个 SK tile（tile 0 ~ tile 199）

#### SK block 的初始化（`gemm()` 行 1069–1077）

```cpp
tile_idx = get_sk_tile_idx(block_iter_end - 1);
// = (block_iter_end - 1) / 256
```

block 0 的 `block_iter_end = 475`，
`tile_idx = (475-1) / 256 = 1`  →  从 **tile 1** 开始

#### SK block 的 tile 遍历（反向）

以 block 0 为例（iters [0, 475)，覆盖 tile 0 和 tile 1 的部分）：

```
第 1 轮：tile_idx = 1
  k_iter_begin(in tile) = iter_begin - tile_iter_begin = 0 - 256 = 0? 不对，需修正：

  tile_iter_begin = 1 × 256 = 256
  iter_begin_for_tile = max(block_iter_begin=0, tile_iter_begin=256) = 256
  k_iter_begin = 256 - 256 = 0  (tile 内起始 K-iter)
  k_iter_end   = 475 - 256 = 219
  k_iters_remaining = 219

  → k_begin = 0 × 16 = 0，k_end = 219 × 16 = 3504
  → 处理 tile 1 的 K=[0, 3504) 这一段
  → tile_started() = (iter_begin == tile_iter_begin) = (256==256) = true  ← 这个 block 是 tile 1 的"第一个" block
  → tile_finished() = (k_iter_end == iters_per_tile) = (219==256) = false ← 没做完
  → 将 partial accumulator 写入全局 workspace（share_accumulators）

  block_iters_remaining = 475 - 219 = 256 → 不为 0，继续

第 2 轮：tile_idx = 0（tile_idx--）
  tile_iter_begin = 0 × 256 = 0
  iter_begin_for_tile = max(0, 0) = 0
  k_iter_begin = 0 - 0 = 0
  k_iter_end   = (0 + 256) - 0 = 256
  k_iters_remaining = 256

  → k_begin = 0，k_end = 4096（完整 tile）
  → tile_started() = (0 == 0) = true  ← 第一块
  → tile_finished() = (256 == 256) = true ← 做完了
  → 直接执行 epilogue，写出 C[tile 0 的输出]

  block_iters_remaining = 256 - 256 = 0 → 退出
```

**谁负责完成 tile 1？**

tile 1 需要迭代 [256, 512)，其中 [256, 475) 由 block 0 覆盖（但 block 0 是 tile_started=true 的"第一块"，没做完），剩余 [475, 512) 由 block 1 的某一轮覆盖。

block 1 的迭代范围 [475, 950)，`tile_idx = (950-1)/256 = 3`，所以 block 1 先处理高 tile（tile 3、tile 2），最终轮处理 tile 1 的 [475-256, 512-256) = [219, 256) 这 37 步，此时：
- `tile_started() = (max(475, 256) == 256) = (475 != 256) = false` ← 不是第一块
- `tile_finished() = (k_iter_end=256 == iters_per_tile=256) = true` ← 是最后一块
→ block 1 在完成 tile 1 时调用 `acquire_accumulators()`，从 workspace 加载 block 0 的 partial，求和后写出 C

---

### DP Block 的详细执行过程

#### 初始化（`gemm()` 行 1041–1067）

```cpp
int dp_block_idx = block_idx - dp_start_block_idx;  // = block_idx - 108

// 第一 DP wave（dp_block_idx < avail_sms=108）：
tile_idx = sk_tiles + dp_block_idx = 200 + dp_block_idx
tile_allottment = dp_first_wave_tiles = 1

// 第二 DP wave 及以后（dp_block_idx >= 108）：
tile_idx = 200 + dp_block_idx + (dp_first_wave_tiles-1) × 108 = 200 + dp_block_idx
tile_allottment = 1
```

#### DP block 的 tile 遍历（正向跨 stride）

以 block 108（`dp_block_idx=0`）为例：

```
初始 tile_idx = 200 + 0 = 200
block_iters_remaining = 256 × 1 = 256

第 1 轮：处理 tile 200（完整 K=[0,4096)）→ 执行 epilogue 写 C
  block_iters_remaining = 256 - 256 = 0 → 退出

以 block 109（dp_block_idx=1）为例：
  第 1 轮：tile 201 → 写 C → 退出
...
以 block 215（dp_block_idx=107）为例：
  第 1 轮：tile 307 → 写 C → 退出
以 block 216（dp_block_idx=108，第二 wave）：
  第 1 轮：tile 200+108 = 308 → 写 C → 退出
...以此类推，直到 tile 1279
```

所有 DP 块 **k_begin=0, k_end=K=4096**，不需要 partial accumulator，执行完直接写出结果。

---

### C 矩阵 Tile 的遍历顺序（重点）

#### Tile 线性编号 → 矩阵坐标

由 `get_tile_offset(tile_idx)` 决定（`threadblock_swizzle_streamk.h` 行 661–690）：

本例 `tiled_shape_m=40, tiled_shape_n=32`，由于 `tiled_shape_m(40) > tiled_shape_n(32)`，
使用 **row-major 光栅**（先沿 N 列方向增加，再沿 M 行方向增加）：

```
m = tile_idx / tiled_shape_n = tile_idx / 32
n = tile_idx % tiled_shape_n = tile_idx % 32
```

| tile_idx | m | n | 对应 C 矩阵的输出区域 |
|----------|---|---|----------------------|
| 0        | 0 | 0 | C[0:128, 0:128]      |
| 1        | 0 | 1 | C[0:128, 128:256]    |
| 2        | 0 | 2 | C[0:128, 256:384]    |
| ...      |   |   |                      |
| 31       | 0 | 31| C[0:128, 3968:4096]  |
| 32       | 1 | 0 | C[128:256, 0:128]    |
| 33       | 1 | 1 | C[128:256, 128:256]  |
| ...      |   |   |                      |
| 1279     | 39| 31| C[4992:5120, 3968:4096] |

**简而言之：先遍历所有列（N 方向），再换下一行（M 方向）。**  
即 (0,0) → (0,1) → ... → (0,31) → (1,0) → (1,1) → ... → (39,31)，正好是 C 矩阵的行主序扫描。

---

### 对于给定 Tile (m, n)，A/B 矩阵的访问范围

每个 tile 计算的是 `C[m*128:(m+1)*128, n*128:(n+1)*128]`，需要：

- **矩阵 A（RowMajor，形状 5120×4096）**：访问 `A[m*128:(m+1)*128, :]`，即 A 的第 m 个"行块"，形状 128×4096
  - 每次 K-tile 步进访问：`A[m*128:(m+1)*128, k_step*16:(k_step+1)*16]`，形状 128×16
  - K 从 0 遍历到 255（共 256 步），从左到右遍历 A 的所有列
  
- **矩阵 B（ColumnMajor，形状 4096×4096）**：访问 `B[:, n*128:(n+1)*128]`，即 B 的第 n 个"列块"，形状 4096×128
  - 每次 K-tile 步进访问：`B[k_step*16:(k_step+1)*16, n*128:(n+1)*128]`，形状 16×128
  - K 从 0 遍历到 255，从上到下遍历 B 的所有行

**一个具体的例子**：tile_idx=33，即 (m=1, n=1)：
```
A 的访问范围：A[128:256, 0:4096]  (整整一个 128 行高的水平带)
B 的访问范围：B[0:4096, 128:256]  (整整一个 128 列宽的垂直带)
输出：        C[128:256, 128:256]
```

对一个 128×128 的 C tile 而言，每个 K-step（16 宽）：
- 加载 A 的 128×16 子矩阵（当前行块的当前 K 列段）
- 加载 B 的 16×128 子矩阵（当前 K 行段的当前列块）
- 执行 128×128×16 的 GEMM，结果累加到 accum

256 步做完后，accum 包含完整的 C[128:256, 128:256] 的结果。

---

### SK tile（0~199）vs DP tile（200~1279）的物理位置

由于 tile 编号按行主序排列，SK 负责的 tile 0~199 在 C 矩阵中是：

```
tile 0~31:   行 m=0 的全部 32 个列块 → C[0:128, 0:4096]
tile 32~63:  行 m=1 的全部 32 个列块 → C[128:256, 0:4096]
...
tile 192~199: 行 m=6 的前 8 个列块  → C[768:896, 0:1024]

tile 200 开始：行 m=6 的第 9 个列块 → C[768:896, 1024:...]（DP 负责）
```

SK 块集中处理 C 矩阵的左上角约 7 行（前 200 个 tile），由 108 个持久化 SK block 分工合作完成。DP 块以标准数据并行方式处理剩余的 1080 个 tile（每 SM 约 10 个 tile）。

---

### 完整执行流程图（以 block 0 为例）

```
block 0 (SK, big, iters [0, 475))
  │
  ├─ 初始化：tile_idx = get_sk_tile_idx(474) = 474/256 = 1
  │
  ├─ 第 1 轮：tile 1，(m=0, n=1)，k=[0, 3504)（219 个 K-step）
  │   ├─ init_iterator_A: ptr_A + 0*128行 * stride_A，偏移到 k_begin=0
  │   ├─ init_iterator_B: ptr_B + 1*128列 * stride_B，偏移到 k_begin=0
  │   ├─ 执行 MmaMultistage(219 iterations)
  │   │   ├─ prologue: 预取 K-step 0,1,2 → smem stage 0,1,2
  │   │   └─ 主循环: K-step 3→4→...→218，每步消耗一个 smem stage 并预取下一个
  │   ├─ tile_finished=false → share_accumulators（写入 workspace）
  │   └─ block_iters_remaining = 475-219 = 256
  │
  └─ 第 2 轮：tile 0，(m=0, n=0)，k=[0, 4096)（完整 256 个 K-step）
      ├─ init_iterator_A: ptr_A + 0行 * stride，k_begin=0
      ├─ init_iterator_B: ptr_B + 0列 * stride，k_begin=0
      ├─ 执行 MmaMultistage(256 iterations)
      ├─ tile_started=true，tile_finished=true
      └─ do_epilogue（写出 C[0:128, 0:128]）→ 退出
```

---

## SK Blocks 与 DP Blocks 的实际运行顺序

### 结论：并发执行，无先后顺序

SK blocks 和 DP blocks **不存在谁先谁后的顺序**，它们是同一次 kernel launch 的所有 block，由 GPU 调度器并发调度到各个 SM 上运行。

### 为什么是并发？

`gemm()` 是一个 CUDA device 函数，整个 grid（本例共 1188 个 block）在同一次 kernel launch 中启动。每个 block 在 `gemm()` 入口处读取自己的 `block_idx`，然后各自走进 `if (dp_block)` 或 `else if (sk_block)` 分支，彼此完全独立地运行。代码中没有任何全局 barrier 在 SK 和 DP 之间做同步：

```cpp
// gemm_universal_streamk.h, 行 1018–1126
void gemm() {
    int block_idx = params.block_mapping.get_block_idx();
    bool dp_block = (block_idx >= dp_start_block_idx) && (...);
    bool sk_block = (block_idx < sk_padding_start_block_idx);

    if (dp_block) {
        // DP block：直接初始化，开始跑自己的 tile
        ...
    } else if (sk_block) {
        // SK block：直接初始化，开始跑自己的迭代段
        ...
    }

    while (true) {
        process_tile(...);     // SK 和 DP 各自在这里工作
        if (iters_done) break;
        ...
    }
}
```

GPU 硬件调度器会根据 SM 空闲情况将这 1188 个 block 分批上机，**block_idx 较小并不意味着先执行**——调度顺序由硬件决定，与 block_idx 大小无关。

### 那 SK blocks 之间如何同步？

SK blocks **内部**之间确实有依赖：一个"non-finishing"SK block（`tile_finished=false`）将 partial accumulator 写入全局 workspace，随后某个"finishing"SK block（`tile_finished=true`）需要读取这些 partial 并做归约，才能最终写出 C。

这个依赖通过 `share_accumulators` / `acquire_accumulators` 中的**原子标志位 + 自旋等待**来实现，而不是靠 block 调度顺序：

```
non-finishing SK block:        finishing SK block:
  compute partial accum           compute its own partial
  write to workspace              spin-wait for peer flag == ready
  set flag = ready          →     read workspace partials
                                  reduce + write C
```

即便 finishing block 比 non-finishing block 更早被调度上 SM，它也会在自旋等待阶段停下来，直到所有 peer 完成写入。

### DP blocks 是否依赖 SK blocks？

**完全不依赖。** DP blocks 处理的是 tile 200~1279，SK blocks 处理的是 tile 0~199，两者操作的是 C 矩阵的不同区域，输入也各自独立读取全局 A/B，之间没有任何数据依赖。DP blocks 从启动的第一刻就可以直接开始工作，无需等待任何 SK block 完成。

### 实际时序示意图

```
时间 →

SM 0:   [SK block 0: tile 1 partial] [SK block 0: tile 0 finish → write C]
SM 1:   [SK block 1: tile 3 partial] [SK block 1: tile 2 partial] [SK block 1: tile 1 finish]
...
SM 107: [SK block 107: ...]
SM 108: [DP block 108: tile 200 → write C] [DP block 108: tile 308 → write C] ...
SM 109: [DP block 109: tile 201 → write C] ...
...
```

SK block 0 在等待来自其他 SK block 的 partial 时（自旋），SM 0 的线程会持续轮询内存中的标志位（occupancy=1 时无法被别的 block 抢占）。DP blocks 同期在各自的 SM 上独立运行，互不干扰。

---

## 更有利于通信 overlap 的计算顺序

### 问题的本质：计算/通信比（Arithmetic Intensity per Fetch）

设矩阵沿每个维度各有 n 个 tile（M=N=K=n，方阵简化分析）：

**当前方案（output-tile-stationary，输出 tile 为内层固定）：**

| 指标 | 值 |
|------|-----|
| 每个 C tile 的计算量 | 1/n³ of total |
| 每个 C tile 首次访问的通信量 | 2n 个 tile（整行 A + 整列 B）= 2/n of total |
| 计算/通信比 | (1/n³) / (2/n) = **1/(2n²)** → n 越大越差 |

每传输一个单位的数据，只能换来极少量的计算——这正是 overlap 很难做的原因。

**目标：找到一种计算顺序，使得每次通信能驱动尽可能多的计算。**

---

### 解决方案：K-slice-stationary（K 切片固定，外层循环为 K）

将当前"output-tile 为外循环、K 为内循环"的顺序翻转，改为**K 为最外层循环**，每个 K-slice 处理完所有输出 tile 的对应部分再推进：

```
当前顺序（output-tile-stationary）：
  for each output tile C[i,j]:          // 外：C tile
      for k = 0..K/kb-1:               // 内：K 方向
          C[i,j] += A[i,k] * B[k,j]

新顺序（K-slice-stationary）：
  for k = 0..K/kb-1:                   // 外：K-slice
      prefetch A[:,k+1] 和 B[k+1,:]   // ← 预取下一个 K-slice
      for all (i,j):                    // 内：所有 C tile
          C[i,j] += A[i,k] * B[k,j]  //   partial update
  write out all C tiles
```

| 指标 | 值 |
|------|-----|
| 每个 K-slice 的通信量 | n + n = 2n 个 tile（A 的一列 + B 的一行）= 2/n of total |
| 每个 K-slice 的计算量 | n² 个 C tile × 1/n³ = **1/n** of total |
| 计算/通信比 | (1/n) / (2/n) = **1/2** → 与 n 无关，常数！ |

**每传输同样多的数据，做了 n 倍的计算。** 随着矩阵增大，overlap 的收益只会更好，而不是更差。

---

### 直观几何解释

```
C = A × B，其中 A 是 M×K，B 是 K×N

K-slice k 对应的数据：
  A 的第 k 列块（整列，M×kb）：被所有 n 个 C 行块用到
  B 的第 k 行块（整行，kb×N）：被所有 n 个 C 列块用到

每个 K-slice 构成一次"外积更新"（rank-kb update）：
  ΔC[:,:] += A[:,k] ⊗ B[k,:]

→ 一次通信（2n 个 tile）驱动 n² 个 C tile 的 partial update
→ n 越大，每次通信的计算"收益"越高
```

对比当前方案中 C tile (i,j) 的 2n 次通信只驱动 1 个 tile 的计算，K-slice 方案同样的 2n 次通信驱动 n² 个 tile。

---

### Pipeline 结构

```
时间轴 →
通信线程：  [fetch K0]  [fetch K1]  [fetch K2]  [fetch K3] ...
计算线程：              [compute K0] [compute K1] [compute K2] ...
                        ↑ n² tile 的 partial update
```

只需 **深度为 1 的双缓冲**（一个 K-slice 在传输，一个在计算），就可以实现接近 100% 的 overlap——前提是 compute K-slice 的时间 ≥ 传输下一个 K-slice 的时间。由于计算/通信比是常数 1/2（且随 n 增大而改善），只要矩阵不是太小，这个条件是可以满足的。

---

### 代价与 CUTLASS 中的实现方式

**代价：需要暂存所有 C tile 的 partial accumulators。**

- 每个 K-slice 结束后 C tile 尚未完成，必须将 partial sum 写回显存（类似 StreamK 的 workspace 机制）
- 总额外显存：n² × TB_M × TB_N × sizeof(accumulator)，即整个 C 矩阵大小（float32）

**与 StreamK workspace 的关系：**

StreamK 已经有了 partial accumulator 的 workspace 基础设施（`share_accumulators` / `acquire_accumulators`）。K-slice-stationary 顺序可以在这个基础上实现：
- 每一"轮"（round）对应一个 K-slice
- 所有 block 在 K-slice 边界处协调，将 partial 写入 workspace 再读取
- 通信预取在 K-slice 计算期间异步执行

**已有的相关理论算法：**

| 算法 | 核心思想 | 与本方案的关系 |
|------|----------|--------------|
| SUMMA | K-outer loop，broadcast A 列和 B 行 | 本方案的分布式等价 |
| Cannon's | 2D ring shift，每步做一个 K-slice | 与本方案同阶 |
| 2.5D GEMM | 用更多内存进一步减少通信 | 本方案的增强版 |

这些算法在分布式内存 GEMM（MPI 场景）中是标准做法，本质上都是将 K 作为外层维度以最大化计算/通信比。

---

## 文件索引

| 功能 | 文件路径 | 关键行 |
|------|----------|--------|
| 命令行入口与 tile 配置 | `examples/14_ampere_tf32_tensorop_gemm_multigpu_new/ampere_tf32_tensorop_gemm.cu` | 293–320, 618–634 |
| Block 角色分配与 tile 遍历 | `include/cutlass/gemm/kernel/gemm_universal_streamk.h` | 1018–1126 |
| SK 迭代范围分配 | `include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h` | 754–778 |
| SK tile_idx 到 (m,n) 映射（光栅化） | `include/cutlass/gemm/threadblock/threadblock_swizzle_streamk.h` | 661–690 |
| 全局内存迭代器（cp.async 驱动） | `include/cutlass/transform/threadblock/predicated_tile_iterator.h` | 241–395 |
| ThreadBlock GEMM 主循环 | `include/cutlass/gemm/threadblock/mma_multistage.h` | 1004–1041 |
| Prologue：smem 预取 | `include/cutlass/gemm/threadblock/mma_multistage.h` | 585–705 |
| 主循环 + 内层 warp MMA | `include/cutlass/gemm/threadblock/mma_multistage.h` | 764–894 |
| smem 写 stage 推进 | `include/cutlass/gemm/threadblock/mma_multistage.h` | 365–386 |
| smem 读 stage 推进 | `include/cutlass/gemm/threadblock/mma_multistage.h` | 351–360 |
| Warp 分区初始化 | `include/cutlass/gemm/threadblock/mma_multistage.h` | 336–347 |
| smem 布局（环形缓冲区大小） | `include/cutlass/gemm/threadblock/mma_base.h` | 140–196 |
| WarpCount 与 kWarpGemmIterations | `include/cutlass/gemm/threadblock/mma_base.h` | 111–117 |
