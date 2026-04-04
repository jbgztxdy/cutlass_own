# 实验记录
## cutlass运行
使用：
```
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=80
make 14_ampere_tf32_tensorop_gemm -j$(nproc)
./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm 
```
即可编译并运行基于sm80的gemm并统计时间。

## 分析
`/home/yjw/workspace/cutlass/include/cutlass/gemm/kernel/gemm.h`是真正在device上跑的、会翻译成ptx代码的部分，203行operator即真正的gemm触发时的动作。
`cutlass/include/cutlass/gemm/threadblock/mma_multistage.h`中是mma代码，其中708行operator会先将首批迭代需要的数据加载进共享内存，之后进入iterator，有通信——访存重叠。

## 优化
一个gemm kernel跑在一个grid上，一个grid包含多个SM，每个SM轮流跑一个或多个block（cta），现在我们希望在第一个block计算开始之前用访存指令将第一个block需要的数据加载进shared memory，之后开始第一个block的计算，同时异步并行加载第二个block所需的数据；在第一个block计算完成切换到第二个block时，此时已经加载完其所需数据，直接开始计算，同时异步并行加载第三个block所需数据，依次类推……直到最后一个block计算完成。如此，则实际上访存站用的时间只有第一个block访存那一小段，后面的都被计算覆盖了（当然，我们应该尽量使用不占用SM core的异步加载指令，直接将其从global memory搬运到shared memory而过程中不占用寄存器这些资源，否则还是影响计算，没有实际上提高性能）。
又由于虽然每个SM有自己的Shared Memory，但Shared Memory 的生命周期与物理上的线程块（CTA）强绑定，第一个cta没办法加载后面cta需要的数据到共享内存。因此我们采用持久化内核的方式来实现通信重叠。新增`cutlass/examples/14_ampere_tf32_tensorop_gemm_new`测试，使用持久化内核，实现了性能提高（可用`--persistent-blocks`参数设置SM数目，可用`--tile-config`设置tile粒度，设为19时最快）。测试指令为：
```
make 14_ampere_tf32_tensorop_gemm_new -j$(nproc)
./examples/14_ampere_tf32_tensorop_gemm_new/14_ampere_tf32_tensorop_gemm_new 
```
最大提升为：
```
yjw@node192:~/workspace/cutlass/build$ ./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm
5120 x 4096 x 4096 TF32 tensor op Matrix Multiply
Runtime: 1.4686 ms
 GFLOPs: 116982
Passed
yjw@node192:~/workspace/cutlass/build$ ./examples/14_ampere_tf32_tensorop_gemm_new/14_ampere_tf32_tensorop_gemm_new --tile-config 19
5120 x 4096 x 4096 TF32 tensor op Matrix Multiply
Runtime: 1.30867 ms
 GFLOPs: 131277
Passed
```
1024.1024.1024下：
```
yjw@node192:~/workspace/cutlass/build$ ./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm --m=1024 --n=1024 --k=1024
1024 x 1024 x 1024 TF32 tensor op Matrix Multiply
Runtime: 0.053312 ms
 GFLOPs: 40281.4
Passed
yjw@node192:~/workspace/cutlass/build$ ./examples/14_ampere_tf32_tensorop_gemm_new/14_ampere_tf32_tensorop_gemm_new --m=1024 --n=1024 --k=1024 --tile-config=13
1024 x 1024 x 1024 TF32 tensor op Matrix Multiply
Runtime: 0.0315392 ms
 GFLOPs: 68089.4
Passed
```

之后测试多GPU上的性能（GPU0上存数据，GPU1取址或访问），发现性能最好的为配置24：
```
(base) yjw@node192:~/workspace/cutlass/build$ ./examples/14_ampere_tf32_tensorop_gemm_multigpu/14_ampere_tf32_tensorop_gemm_multigpu --storage-device=0 --compute-device=1
5120 x 4096 x 4096 TF32 tensor op Matrix Multiply
Storage GPU: 0, Compute GPU: 1
Note: Kernel will access data remotely via PCIe/UVA
Runtime: 22.6489 ms
 GFLOPs: 7585.29
Passed
(base) yjw@node192:~/workspace/cutlass/build$ ./examples/14_ampere_tf32_tensorop_gemm_multigpu_new/14_ampere_tf32_tensorop_gemm_multigpu_new --tile-config=24 --storage-device=0 --compute-device=1
5120 x 4096 x 4096 TF32 tensor op Matrix Multiply
Storage GPU: 0, Compute GPU: 1
Note: Kernel will access data remotely via PCIe/UVA
Runtime: 15.5834 ms
 GFLOPs: 11024.5
Passed
```
加入大缓存：
```
(base) yjw@node192:~/workspace/cutlass/build$ ./examples/14_ampere_tf32_tensorop_gemm_multigpu_new/14_ampere_tf32_tensorop_gemm_multigpu_new --tile-config=12 --storage-device=0 --compute-device=1 --software-cache=1
5120 x 4096 x 4096 TF32 tensor op Matrix Multiply
Storage GPU: 0, Compute GPU: 1
Note: Kernel will access data remotely via PCIe/UVA
Runtime: 5.73711 ms
 GFLOPs: 29945.1
Passed
```