# A–E 待验证方案（未实施）

用户于 2026-09-09 确认：本次只打包当前代码供别人研究，保留以下方案，不直接修改 kernel。

## A：增大 hIn 行块和 D 分片

提案：`kPreBlockRows=8` 保持；`kHinBlockRows: 4→8`；`kHinTileElements: 256→1024`；`kSumBatchRows=64` 保持。目标是减少大 D 的分片循环，并以一个 hIn 行块覆盖一个后处理组。

阻断条件：不能直接套用现有缓冲分配。`hInXQueue` 的每块形状是 `[rows,n,tile]`，不是 `[rows,tile]`。在 `rows=8,n=8,tile=1024`、输入 FP16/BF16、双缓冲下，仅这一队列就需 `2*8*8*1024*2=262144 B=256 KiB`；对应 FP32 `hInXCastBuf` 还需 `8*8*1024*4=262144 B`。尚未计入前处理、输出、scratch 等空间，已经超出项目采用的 192 KiB UB 预算。“总共约 168KB”的提案计算漏掉了 stream 维度。

因此需要先重新设计缓冲复用、按 shape 自适应行块/分片或分阶段分配，再实施 A；不能仅改两个常量。

## B：合并 LoadHInBlock 输入 DMA

提案：将逐行 `blockCount=n` 的 DataCopyPad 合并为一次 `blockCount=rows*n`；`blockLen=current*sizeof(DT_X)`，`srcStride=(headDim-current)*sizeof(DT_X)`，`dstStride=0`，起点 `firstRow*flatDim+dimOffset`，输出排列 `[rows,n,current]`。

地址依据：`flatDim=n*headDim`，块编号 `i=r*n+s` 的源地址等于 `firstRow*flatDim+i*headDim+dimOffset`。在当前对齐条件下可减少 DMA 调用数，但调用数不等于实际总线事务数，不能承诺同倍数提速。接手者应先查看当前 LoadHInBlock，避免重复实施已经存在的合并逻辑，并验证尾块、队列生命周期和所有 n。

## C：逐行/组化双路径和有效 lane

提案：`rowCount<8 || n!=8` 时走 `PostprocessRowsSmall`；其余走组化路径；每组 Sigmoid 长度从固定 64 改为 `groupRows*8`。提案使用已展开的 bias 区域 `bias[80]`、`bias[144]` 与常量 1 区域 `bias[208]`，Sigmoid 的 scale 设为 1。

必须核实这些区域的容量、初始化，以及 alpha 是否已在上游乘入，防止重复缩放或遗漏缩放。小路径目前仅是设计意图，不是完整实现：须保证 pre、post、res 三段都被正确加载和计算，hIn 使用正确权重，尾部输出完整，同步与初始化齐全。提案中的“每行一次 n 加一次 res”不能在未说明 pre/post 布局时认定覆盖了全部输入。

`n=4/6` 是否应全部逐行、实际测试点每核是否只有 1–3 行，均应通过真实 shape 与测量确认。不能把 lane 数减少直接等同于向量指令条数同比减少。

## D：调整 WarmPostprocessInputs

提案：保留 alpha/bias/one 初始化与必要同步，移除屏障前首块 hIn 预取，将首块加载放到屏障后，与 hMix 加载协调。

必须同时修改后处理首次 DeQue 的生产者路径，避免删除 Warm 预取后队列为空。小路径同样需要自己的初始化和首次入队。移除预取是否改善性能尚未证实，应与其他变化隔离测试；不能无证据认定 AIV MTE2 与 AIC matmul 使用同一条本地搬运流水线。

## E：审计与版本绑定

提案：审计常量同步到实际实现；重新检查总 UB、每个缓冲的最大访问范围、GM/UB 步长、尾块、广播、队列生命周期以及小路径完整性。

审计不能通过豁免 redline 或修改独立常量来替代对真实源码的检查。打包前应绑定所有提交文件哈希与远端编译产物、任务 ID；探针保持 0。

## 实验约束

一次同时更改 A–E 是一个组合实验，不是单变量实验。若目标是识别收益来源，先保留明确 Pass 基线，再分别测试 DMA 合并、有效 lane、双路径、Warm 调整和缓冲方案；若组合测试获得收益，只能证明组合有效，不能归因到其中某一步。

用户提出的“小 shape 下降 3–5 μs 后保留”“n=8 尝试 padNone”均为待检验的预期，不是本次实测结果。改变 padding 时须继续满足对齐和初始化要求。
