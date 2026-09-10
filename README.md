# MhcPre_group_vec_v1

> 当前修复分支请先读 [审阅入口](REVIEW_START_HERE.md) 和
> [边界修复与版本绑定说明](docs/PREPROCESS_BOUNDS_REVIEW.md)。
> 下文保留旧版本说明，其中 tile=1024、槽位布局和 UB 数字已过时。
> 当前 host tile=512、hIn tile=256；本分支尚无 CANN/NPU 验证结果。

组级向量后处理候选（batch_v1 的直通继承者）。在保持官方单轮 8/9 协议、
浮点顺序与 host M64 tiling 完全不变的前提下，把 AIV 后处理从“逐行标量 +
逐 stream 循环”改成“组(8 行)级聚合 DMA + Brcb 批量归约”。

## 相对 batch_v1 的改动（全部在 `code/op_kernel/mhc_pre.cpp`）

1. **队列资源合法化**：所有 `TQue` 深度恒为 2（`kQueueDepth`），移除非法的
   `kXQueueDepth=17`。
2. **组级 hMix 装载** `LoadMixGroup`：每组 8 行用 3 次跨步 `DataCopyPad`
   （pPre/pPost/pRes 各一次，blockCount=rows）替代 24 次逐行小片搬运；
   双组槽位交替预取，DMA 与计算重叠。
3. **组级 hPost/hRes 输出** `CopyGroupOutput`：每组各 1 次
   `DataCopyPad`（blockCount=groupRows，src 在 96 宽槽位、GM 侧行连续），
   替代 16 次逐行小片拷贝。
4. **Brcb 批量 hIn**：自官方 ops-transformer 原样移植
   `MulABLastDimBrcInline2` + `ReduceSumARAPerf`（batchY 配置已在评测机
   通过）。逐 (row, tile) 一次聚合 DMA 搬 n 个 stream → 一次 Cast →
   一次广播乘 → 一次跨 n 归约，消除逐 stream 标量往返。
   x 分片沿 D 的上限 `kHinTileElements=1024`，控制 UB 占用。
5. `headDim % 16 == 0`（host 已保证）使每个 hIn tile 长度均为 16 的倍数，
   Brcb 的行跨距恒等于 tile 长度。

## 布局约定

- hMix/bias 三段(pPre[n] / pPost[n] / pRes[n*n])各自落 32B 对齐槽位：
  每行 96 float（`kSlotPost=8`, `kSlotRes=16`, `kSlotBufElements=96`, n<=8）。
- 组级输出缓冲 `floatOutQueue_`：`kPostGroupRows x 96` floats，
  hPost 在 +kSlotPost、hRes 在 +kSlotRes。
- hPre 组槽 `[8 行 x 16]`（8 流 + 填充），供 Brcb 直接按 block 消费。

## UB 预算（n=8, vectorRows=5472, hInTile=1024）

约 **149,632 B < 192 KiB**（`audit_group_vec.py` 校验）。

## 目录

- `code/` — 提交代码（op_host/op_kernel/CMakeLists）。
- `audit_group_vec.py` — 离线审计：分区覆盖、形状矩阵、位精确公式复算、
  UB 预算、队列深度 <= 2、红线（协议/浮点顺序/直接 Matmul 面）。
- `cloud_runner/` — 云端递进验证：
  - `probe_e1e2.cpp` — E1（8/9 协议 + 首次 IterateAll）、E2（同 executor
    二次启动）。
  - `runner.cpp` — E3 全形状矩阵正确性 + 时延（fp16/bf16 x gamma on/off）。
  - `run_all.sh` — 一键 cmake/make + 编译 runner + E1/E2/E3。

## 云端验证

```bash
# 把本目录同步到云端后：
bash cloud_runner/run_all.sh
```

E1/E2/E3 全绿后运行 `python audit_group_vec.py` 做最终红线复核，再打包。

## 本地审计

```bash
python audit_group_vec.py
```
