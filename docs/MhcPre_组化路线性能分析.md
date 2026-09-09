# MhcPre 组化路线性能分析

## 总体判断

18 μs 这个结果目前不能直接归因于 V3。根据现有事实，V3 同时存在源码与结果未绑定、host `TILE=2048` 与 UB 预算冲突、xGamma DMA 调用形式存疑等问题。

E8 还曾存在 bias 越界、`one` 越界和非法 DMA 组合。如果这些代码真的运行，通常不会同时得到稳定的 0% 错误。因此，18.28 μs / 19.86 μs 更可能来自旧版本或某个中间二进制。只有拿到带源码 hash 的可编译基线后，才有意义分析运行时退化。

## 1. 退化根因排序

如果确认远端确实运行了 V3，建议按以下顺序排查：

1. **实际行数太少，组化无法摊薄固定开销。**

   每个 AIV 的行数是 `ceil(BS / 48)`。榜单测试点很可能是小 shape，若每个 AIV 只有 1 至数行，8 行一组就无法发挥优势。

2. **核对组级 Sigmoid 是否固定处理 `count=64`。**

   如果仍固定处理 64 lane，而实际有效元素只有 `activeRows * n`，小 `rowCount` 会产生大量无效计算。`n=4/6` 时固定 8 槽还会进一步放大工作量。应改为有效元素数量，或对小行数走逐行路径。

3. **AIV 内新增队列和同步造成串行化。**

   `WarmPostprocessInputs`、多次 `EnQue/DeQue`、UB→UB 拷贝和组间预取都有固定成本。小 workload 中，这些成本可能超过节省的向量指令。

4. **`LoadHInBlock` 的 DMA 描述符数量没有真正减少。**

   如果当前实现对每一行发一个 `blockCount=n` 的 2D DMA，数据量虽然不变，但提交和描述符开销仍然存在。合并成一次 `blockCount=rows*n` 的 DCP 才可能明显降低事务数。

5. **`LoadMixGroup` 的三次跨步搬运和 padding 成本。**

   对 `mixDim <= 80` 的小数据段，`rightPadding`、32B 对齐和 Ext DMA 的固定成本可能超过逐行简单搬运节省的操作。

6. **`TILE=2048` 带来的 UB 或资源压力。**

   按已有预算，`xQueue + xCastQueue` 已接近或超过 192 KB。若 2048 确实生效，可能导致 InitBuffer 失败、路径变化或 DMA 无法与计算重叠。

7. **xGamma 的 DMA 重载不明确。**

   `DataCopyPad(Gm, Lm, gParams, padNone)` 能编译，不代表它匹配了预期的 MTE3 Ext 版本。需要根据实际头文件候选和编译器诊断确认第四个参数对应的重载及其语义。单次 xGamma 写出通常不太可能独立造成 9 倍退化，但必须先排除。

如果跨核屏障数量和粒度确实没有增加，它不太可能单独解释 9 倍退化；不过新增的 AIV 内部同步仍可能是主要因素。

## 2. 没有 Git 时的版本绑定

每次提交前生成内容寻址快照，记录：

- host、kernel、audit、构建脚本和 tiling 配置的文件 hash；
- CANN/toolchain 版本和完整编译命令；
- `TILE`、`kHinTileElements`、队列深度等常量；
- 提交包 hash 和编译产物 hash；
- 远端任务 ID、提交时间、正确性和时延。

建议目录结构：

```text
snapshots/<source_sha>/
submissions/<seq>_<source_sha>/
```

源码中加入由脚本生成的 `BUILD_ID`。调试版本可把 magic 值写入独立 debug workspace 槽，由 host 读回确认远端确实使用了新包。若平台支持编译日志，一次带唯一 `#error BUILD_CANARY_xxx` 的编译探针是最强确认方式，之后恢复正式代码。

长期也可以在当前目录执行 `git init`，但规范化文件清单加 SHA-256 已足够解决追溯问题。回退时复制指定快照，不要依赖目录状态或手工记录。

## 3. 没有 profiler 时的定位

采用每次只改一个变量的编译变体，并保持合法的屏障、输出和数据依赖：

- V0 逐行后处理；
- 组化但关闭 Warm 预取；
- 比较逐行 HIn DMA 与单次合并 DMA；
- 比较固定 64 lane 与 `activeRows*n`；
- 比较不同 `rowCount` 和 `n` 的路径。

不能简单把 `WarmPostprocessInputs` 设为空后继续读取未初始化 UB。应使用确定性填充值替代计算，并保留相同同步协议。

如果 CANN 8 头文件提供设备 cycle counter，可在 debug-only 版本记录以下阶段边界：

- Pre 完成；
- hMix 装载完成；
- Sigmoid 完成；
- HIn 完成；
- 输出写回完成。

常见 API 名称是 `GetSystemCycle()`，但应以本地 CANN 8 头文件为准。每个 AIC/AIV core 使用独立 workspace 槽，阶段耗时取所有核心中的最大值。计时边界要先确认 DMA/V pipeline 已完成，并用空 marker 版本校准开销。

`DumpTensor` 只用于抽样检查布局和正确性，不能用于性能测量，因为它会改变 DMA 和同步行为。host 侧应预热后多次运行，交错执行 A/B 版本，比较 median 和离散度，而不是依赖单次 2 μs 数字。

## 4. 组级路线是否值得继续

官方逐 stream 路线已经达到约 2 μs，说明这个算子很可能由固定开销主导，而不是算术吞吐主导。优先级应是：

1. 减少跨核事件和 TQue 同步；
2. 减少 DMA 描述符和事务次数；
3. 再减少向量指令数量。

建议按 `rowCount` 做双路径：

- `rowCount < 8`：使用 V0 逐行路径；
- `rowCount >= 8` 且 `n=8`：启用组化路径；
- `n=4/6`：先验证密集打包，否则继续逐行实现。

组化只有在不增加同步事件、同时显著减少 DMA 事务或确实扩大向量长度时才值得保留。`LoadHInBlock` 单 DCP 应作为独立提交测试，不能和所有正确性修复及其他优化绑定在同一轮。

## 5. 建议的提交顺序

第一轮只做：

1. host `kVectorTileElements = 512`；
2. 使用已确认合法的 xGamma 写出 API；
3. 修复 biasArr 和 oneArr 的布局及越界；
4. 恢复标量 `invRms * alpha` 的舍入顺序；
5. 同步更新审计脚本中的 UB 预算、尾组规则和红线锚点。

拿到带源码 hash 的正确性基线后，第二轮单独测试 HIn 单 DCP 合并。若小 shape 仍明显慢，优先启用逐行快路径，再评估 res bias 组化。

