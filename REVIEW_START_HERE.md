# MhcPre 边界修复审阅入口（2026-09-10）

本分支修复预处理尾块预取越界，以及短 headDim 下共享 scratch 容量不足。
审阅内容、离线验证范围和版本记录流程见 [PREPROCESS_BOUNDS_REVIEW.md](docs/PREPROCESS_BOUNDS_REVIEW.md)。

当前 code/ 已有修改；下面的哈希、打包状态和测试报告属于 2026-09-09 的历史快照。
S1_submission.zip、s1_package/ 与旧 SHA256_MANIFEST.txt 不代表本分支源码。
本次未进行 CANN 编译或 NPU 评测，A/C/B/D 性能实验尚未启用。

## 历史快照（2026-09-09）

本快照用于交给其他开发者分析优化方案，不是新优化版本，也不是本次已验证的比赛提交。

## 当前版本

- kernel SHA-256：`C6C15246CFAB2BA29E8E66335324B22F1654AE78549F0D07549D513BBACC4E31`
- host SHA-256：`379E51463196915A4681D068CE91FAA53E2A3B2CC2271613C2CA4F476A461E06`
- `kTimingProbeIters=0`，本次保持禁用，没有重新启用探针。
- host `kVectorTileElements=512`；kernel `kHinBlockRows=4`、`kHinTileElements=256`。
- 打包时 `code/` 与 `s1_package/code/` 的 kernel 哈希一致。
- 本次未修改计算代码，未运行 NPU、云端构建或比赛评测；未将 offline audit 当作设备验证。

## 建议阅读顺序

1. `code/op_kernel/mhc_pre.cpp` 与 `code/op_host/mhc_pre.cpp`：当前真实实现。
2. `docs/PROPOSED_A_E.md`：用户提出、尚未实施的优化方案及需要验证的前提。
3. `docs/MhcPre_组化路线性能分析.md`：用户提供的原始分析文档，原样保留。它讨论的部分旧版本问题不代表当前版本仍然存在。
4. `audit_group_vec.py`、`cloud_runner/`：已有检查与构建工具；请先审查再运行。

## 测试信息的证据边界

用户报告探针版本为 38.16/36.70 μs，对照为 19.6/21.1 μs，差值约 18.56/15.60 μs，并报告约 ±2–3 μs 的波动。本次没有取得原始评测日志、任务 ID、编译产物哈希或分阶段 profiler 数据。

这些报告支持“注入的设备工作量影响了计时”，但本快照不据此独立认证计时完全排除了 host 开销，也不将“固定开销主导”“小 shape 每核仅 1–3 行”标为已实测事实。接手者应绑定实际 shape、提交记录和编译产物后验证。

## 历史文件和完整性

- 原项目全部非 `.git` 文件保留，包括旧 `README.md`、`S1_submission.zip`、`SHA256_MANIFEST.txt`。旧说明和旧压缩包可能过时，不应覆盖当前 `code/`。
- `s1_package/` 按当前磁盘内容保留，没有重新执行会删除并重建它的打包脚本。
- 审阅 ZIP 附带 `SNAPSHOT_SHA256.txt`，列出包内项目文件的 SHA-256；另附 ZIP 外部校验文件。
- `.git` 管理数据不放入 ZIP；本地 Git 提交另行记录，远端发布需要用户提供目标仓库。

## 环境依赖

该项目依赖 Ascend/CANN 工具链、目标设备以及竞赛或云端构建框架；这些外部依赖不包含在压缩包中。已有脚本使用特定 Linux 路径，打包不代表可在任意目录直接一键编译。特别是 `cloud_runner/run_all.sh` 的工作目录及 CMake 入口应由接手者核对。
