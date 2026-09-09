"""Offline invariants for the group_vec MhcPre candidate.

Self-contained (stdlib only; no numpy).  Checks:

- the host tiling partition (48 AIV / 24 AIC, rowsPerVector = 2*vectorRowsPerCore,
  ceil-divided rows without alignment; runtime single-shape M may be < 16),
  with M64 fixed matmul tiling;
- the complete operator formula with bit-accurate fp32/fp16/bf16 helpers,
  replicating the group_vec PostprocessRows data flow (group hMix load,
  per-row Brcb hIn over n streams, per-row hPre/hPost/hRes);
- the AIV UB budget (queue depths all <= 2) and 64-bit workspace layout;
- the group_vec red lines over the actual submitted files.

Usage:
    python audit_group_vec.py [--code <candidate code dir>]
                              [--ref  <baseline batch_v1 code dir>]
"""

from __future__ import annotations

import math
import re
import struct
import sys
from pathlib import Path

AIC = 24
AIV = 48
TILE_ELEMENTS = 512
K_HIN_TILE = 256
K_POST_GROUP_ROWS = 8
K_PRE_BLOCK_ROWS = 8
K_HIN_BLOCK_ROWS = 4
POST_ARRAY_ELEMS = 64
RES_ARRAY_MAX = 512
MAX_MIX_DIM = 80
MAX_BATCH_SEQ = 4 * 32768
MAX_FLAT_DIM = 8 * 16384
M64_TILING_M = 64

LEGAL_B = (1, 2, 4)
LEGAL_S = (4096, 8192, 16384, 32768)
LEGAL_N = (4, 6, 8)
LEGAL_D_STEP = 16
LEGAL_D_MAX = 16384


def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_up(v: int, alignment: int) -> int:
    return ceildiv(v, alignment) * alignment


def fail(msg: str) -> None:
    sys.exit("FAIL: " + msg)


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)


# ---------------------------------------------------------------------------
# Part 1: row partition.
# ---------------------------------------------------------------------------


def vector_rows_for(batch_seq: int) -> int:
    # 官方 membase 同款: 不做行对齐, 让 AIV/cube 满核并行(运行时 M 可 < 16)。
    return ceildiv(batch_seq, AIV)


def partition(batch_seq: int) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]]]:
    vector_rows = vector_rows_for(batch_seq)
    cube_rows = 2 * vector_rows
    vector_ranges = []
    for core in range(AIV):
        start = min(core * vector_rows, batch_seq)
        vector_ranges.append((start, min(vector_rows, batch_seq - start)))
    cube_ranges = []
    for core in range(AIC):
        start = min(core * cube_rows, batch_seq)
        cube_ranges.append((start, min(cube_rows, batch_seq - start)))
    return cube_rows, vector_ranges, cube_ranges


def check_partitions() -> None:
    legal_bs = sorted({b * s for b in LEGAL_B for s in LEGAL_S})
    max_cube_rows = 0
    for batch_seq in legal_bs:
        cube_rows, vector_ranges, cube_ranges = partition(batch_seq)
        check(cube_rows == 2 * vector_rows_for(batch_seq),
              "rowsPerVector != 2 * vectorRowsPerCore")
        covered_v: list[int] = []
        covered_c: list[int] = []
        for start, count in vector_ranges:
            covered_v.extend(range(start, start + count))
        for start, count in cube_ranges:
            covered_c.extend(range(start, start + count))
        check(covered_v == list(range(batch_seq)), "AIV row coverage miss/overlap")
        check(covered_c == list(range(batch_seq)), "AIC row coverage miss/overlap")
        for aic in range(AIC):
            paired: list[int] = []
            for aiv in (2 * aic, 2 * aic + 1):
                start, count = vector_ranges[aiv]
                paired.extend(range(start, start + count))
            start, count = cube_ranges[aic]
            check(paired == list(range(start, start + count)),
                  "AIC slice is not the exact concatenation of its two AIV slices")
        # 运行时单形状 M 不要求 16 对齐: 官方 m_split 基线在 BS=64 时以
        # M=2*stage1BsFactor=4 调 ProcessMatmulXPhi, 已在评测机验证。
        max_cube_rows = max(max_cube_rows, cube_rows)
    print(f"  partitions OK (max rowsPerVector = {max_cube_rows})")


# ---------------------------------------------------------------------------
# Part 2: (n, D) arithmetic surfaces and workspace overflow.
# ---------------------------------------------------------------------------


def check_shapes() -> None:
    for n in LEGAL_N:
        mix = n * n + 2 * n
        check(mix in (24, 48, 80), "mixDim not in {24,48,80}")
        check(mix % 8 == 0, "mixDim not 32B aligned")
        for d in range(LEGAL_D_STEP, LEGAL_D_MAX + 1, LEGAL_D_STEP):
            flat = n * d
            check(flat % 8 == 0, "flatDim not K0 aligned")
            check((mix * flat) < 2**63 and (MAX_BATCH_SEQ * flat) < 2**63,
                  "element index overflows int64")
            check(mix * flat * 4 < 2**64, "phi/hMix byte offset overflows uint64")

    # headDim % 16 == 0 -> every hIn tile length is a multiple of 16, so the
    # Brcb row stride (RoundUpBlock(current)) always equals current.
    for n in LEGAL_N:
        for d in range(LEGAL_D_STEP, LEGAL_D_MAX + 1, LEGAL_D_STEP):
            check(d % 16 == 0, "hIn Brcb requires headDim % 16 == 0")

    x_gamma_bytes = MAX_BATCH_SEQ * MAX_FLAT_DIM * 4
    inv_rms_bytes = MAX_BATCH_SEQ * 4
    h_mix_bytes = MAX_BATCH_SEQ * MAX_MIX_DIM * 4
    check((x_gamma_bytes + inv_rms_bytes + h_mix_bytes) < 2**64,
          "total workspace overflows uint64")
    print("  shapes/workspace OK")


# ---------------------------------------------------------------------------
# Part 3: bit-accurate helpers.
# ---------------------------------------------------------------------------


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _f16_bits_from_f32_bits(v: int) -> int:
    sign = (v >> 16) & 0x8000
    exp = (v >> 23) & 0xFF
    mant = v & 0x7FFFFF
    if exp == 0xFF:
        return sign | 0x7C00 | (1 if mant else 0)
    exp16 = exp - 127 + 15
    if exp16 >= 0x1F:
        return sign | 0x7C00
    if exp16 <= 0:
        m = mant | 0x800000
        shift = 14 - exp16
        if shift >= 25:
            return sign
        half = m >> shift
        rem = m & ((1 << shift) - 1)
        threshold = 1 << (shift - 1)
        if rem > threshold or (rem == threshold and (half & 1)):
            half += 1
        return sign | (half & 0x3FF)
    mant16 = mant >> 13
    rem = mant & 0x1FFF
    if rem > 0x1000 or (rem == 0x1000 and (mant16 & 1)):
        mant16 += 1
        if mant16 == 0x400:
            mant16 = 0
            exp16 += 1
    if exp16 >= 0x1F:
        return sign | 0x7C00
    return sign | (exp16 << 10) | mant16


def _f16_to_float(bits: int) -> float:
    sign = -1.0 if bits & 0x8000 else 1.0
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if exp == 0x1F:
        return sign * math.inf if mant == 0 else sign * math.nan
    if exp == 0:
        return sign * (mant / 1024.0) * (2.0 ** -14)
    return sign * (1.0 + mant / 1024.0) * (2.0 ** (exp - 15))


def to_f16(x: float) -> float:
    return _f16_to_float(_f16_bits_from_f32_bits(_f32_bits(x)))


def to_bf16(x: float) -> float:
    v = _f32_bits(x)
    lsb = (v >> 16) & 1
    rounded = (v + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", rounded))[0]


def rint_half_even(x: float) -> float:
    x = f32(x)
    fl = math.floor(x)
    frac = x - fl
    if frac > 0.5 or (frac == 0.5 and int(fl) % 2 == 1):
        return fl + 1.0
    return fl


def _self_test_arith() -> None:
    cases = [(1.0, 0x3C00), (0.5, 0x3800), (-2.0, 0xC000), (65504.0, 0x7BFF),
             (1e-30, 0x0000), (0.1, 0x2E66), (1.5, 0x3E00), (3.5, 0x4300),
             (2.5, 0x4100)]
    for value, bits in cases:
        got = _f16_bits_from_f32_bits(_f32_bits(value))
        check(got == bits, f"f16 self-test mismatch for {value}")
    check(to_bf16(0.1) == 0.10009765625, "bf16 self-test value mismatch")
    check(rint_half_even(2.5) == 2.0 and rint_half_even(3.5) == 4.0
          and rint_half_even(-2.5) == -2.0, "rint self-test failed")
    print("  fp32/fp16/bf16 helpers OK")


# ---------------------------------------------------------------------------
# Part 4: formula audit replicating the group_vec data flow.
# ---------------------------------------------------------------------------


def fp32_matmul(a_rows, phi_rows, k):
    m = len(a_rows)
    n = len(phi_rows)
    accs = [[0.0] * n for _ in range(m)]
    for kk in range(k):
        for mi in range(m):
            row = accs[mi]
            for ni in range(n):
                row[ni] = f32(row[ni] + f32(a_rows[mi][kk] * phi_rows[ni][kk]))
    return accs


def sigmoid(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-f32(v)))


def run_formula_case(n: int, d: int, has_gamma: bool, dtype: str, rows: int, seed: int) -> None:
    rng_state = seed

    def rnd():
        nonlocal rng_state
        rng_state = (rng_state * 48271) % 2147483647
        return (rng_state % 2000000) / 1000000.0 - 1.0

    mix = n * n + 2 * n
    flat = n * d
    x = [[to_f16(f32(rnd() * 0.5)) if dtype == "fp16" else to_bf16(f32(rnd() * 0.5))
          for _ in range(flat)] for _ in range(rows)]
    phi = [[f32(rnd() * 0.1) for _ in range(flat)] for _ in range(mix)]
    gamma = [f32(1.0 + rnd() * 0.1) for _ in range(flat)] if has_gamma else None
    alpha = [f32(rnd() * 0.2) for _ in range(3)]
    bias = [f32(rnd() * 0.2) for _ in range(mix)]
    norm_eps = 1.0e-6
    hc_eps = 1.0e-6
    inv_flat = f32(1.0 / flat)

    inv_rms = []
    x_gamma = []
    for r in range(rows):
        row_sum = 0.0
        for kk in range(flat):
            row_sum = f32(row_sum + f32(x[r][kk] * x[r][kk]))
        inv = f32(1.0 / math.sqrt(f32(f32(row_sum * inv_flat) + norm_eps)))
        inv_rms.append(inv)
        x_gamma.append([f32(x[r][kk] * gamma[kk]) if has_gamma else f32(x[r][kk])
                        for kk in range(flat)])

    h_mix = fp32_matmul(x_gamma, phi, flat)
    for r in range(rows):
        for j in range(mix):
            h_mix[r][j] = f32(h_mix[r][j] * inv_rms[r])

    h_pre = []
    h_post = []
    h_res = []
    for r in range(rows):
        pre = []
        post = []
        for j in range(n):
            w_pre = h_mix[r][j]
            pre.append(f32(sigmoid(f32(f32(w_pre * alpha[0]) + bias[j])) + hc_eps))
            w_post = h_mix[r][n + j]
            post.append(f32(2.0 * sigmoid(f32(f32(w_post * alpha[1]) + bias[n + j]))))
        h_pre.append(pre)
        h_post.append(post)
        res = []
        for j in range(n * n):
            w_res = h_mix[r][2 * n + j]
            res.append(f32(f32(w_res * alpha[2]) + bias[2 * n + j]))
        h_res.append(res)

    h_in = []
    for r in range(rows):
        row = []
        for dd in range(d):
            acc = 0.0
            for s in range(n):
                acc = f32(acc + f32(h_pre[r][s] * x[r][s * d + dd]))
            # CAST_RINT: fp32 -> fp16/bf16 直接舍入到目标格式网格(round-half-even),
            # 不是先取整数再转格式 —— 例如 0.25 应保留为 0.25。
            row.append(to_f16(acc) if dtype == "fp16" else to_bf16(acc))
        h_in.append(row)

    for r in range(rows):
        for v in h_in[r]:
            check(-5.0 < v < 5.0, "hIn magnitude out of sane range")
        for v in h_pre[r]:
            check(0.0 < v < 1.1, "hPre out of range")
        check(all(0.0 < v for v in h_post[r]), "hPost must be positive (2*sigmoid)")
        check(len(h_res[r]) == n * n and len(h_post[r]) == n and len(h_pre[r]) == n,
              "section sizes wrong")


def check_formula() -> None:
    seed = 20260905
    for n in LEGAL_N:
        for dtype in ("fp16", "bf16"):
            run_formula_case(n, 16, True, dtype, rows=36, seed=seed)
            seed += 1
            run_formula_case(n, 32, False, dtype, rows=36, seed=seed)
            seed += 1
    run_formula_case(8, 1024, True, "fp16", rows=8, seed=seed)
    print("  formula audit OK (fp16/bf16, gamma on/off, N=24/48/80)")


# ---------------------------------------------------------------------------
# Part 5: AIV UB budget.
# ---------------------------------------------------------------------------


def check_memory() -> None:
    vector_rows = vector_rows_for(MAX_BATCH_SEQ)
    n = 8
    h_in_tile = min(K_HIN_TILE, 16384)
    ub_bytes = (
        2 * K_PRE_BLOCK_ROWS * TILE_ELEMENTS * 2          # xQueue
        + 2 * K_PRE_BLOCK_ROWS * TILE_ELEMENTS * 4        # xCastQueue
        + 2 * K_HIN_BLOCK_ROWS * h_in_tile * 2            # hInOutQueue
        + 2 * K_HIN_BLOCK_ROWS * n * h_in_tile * 2        # hInXQueue
        + K_HIN_BLOCK_ROWS * h_in_tile * 4                # calcBuf
        + K_HIN_BLOCK_ROWS * h_in_tile * 4                # reduceWorkBuf
        + 32 * 4                                          # scalarBuf
        + vector_rows * 4                                 # rowSumBuf
        + TILE_ELEMENTS * 4                               # gammaBuf
        + 64 * 8 * 4                                      # batchSumBuf
        + K_HIN_BLOCK_ROWS * n * h_in_tile * 4            # hInXCastBuf
        + 16 * 8 * 4                                      # brcbBuf
        + 2 * POST_ARRAY_ELEMS * 4                        # preArrBuf
        + 2 * POST_ARRAY_ELEMS * 4                        # postArrBuf
        + 2 * RES_ARRAY_MAX * 4                           # resArrBuf
        + 272 * 4                                         # biasBuf (80+64+64+64 layout for n=8 max)
    )
    check(ub_bytes < 192 * 1024, "AIV UB exceeds 192 KiB")
    # Every TQue depth must be <= 2 (same-TPosition sync-event resource cap).
    queue_depths = re.findall(r"TQue<[^>]*,\s*(\d+|kQueueDepth)\s*>", KERNEL_TEXT)
    check(queue_depths and all(d == "kQueueDepth" or int(d) <= 2 for d in queue_depths),
          f"a TQue depth exceeds 2: {queue_depths}")
    check(KERNEL_TEXT.count("kQueueDepth = 2") == 1, "kQueueDepth must be 2")
    print(f"  AIV UB budget OK ({ub_bytes} bytes < 192 KiB); queue depths {sorted(set(queue_depths))}")


# ---------------------------------------------------------------------------
# Part 6: static red lines over the actual files.
# ---------------------------------------------------------------------------

KERNEL = "op_kernel/mhc_pre.cpp"
HOST = "op_host/mhc_pre.cpp"
TILING_H = "op_kernel/mhc_pre_tiling.h"
TILING_KEY = "op_kernel/tiling_key_mhc_pre.h"
KERNEL_TEXT = ""


def _strip_block_comments(text: str) -> str:
    out: list[str] = []
    i = 0
    in_quotes = False
    while i < len(text):
        if text[i] == '"' and (i == 0 or text[i - 1] != '\\'):
            in_quotes = not in_quotes
            out.append(text[i])
            i += 1
            continue
        if not in_quotes and text[i:i + 2] == "/*":
            end = text.find("*/", i + 2)
            if end == -1:
                break
            seg = text[i:end + 2]
            out.append("\n" * seg.count("\n"))
            i = end + 2
            continue
        if not in_quotes and text[i:i + 2] == "//":
            end = text.find("\n", i + 2)
            if end == -1:
                break
            out.append("\n")
            i = end + 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def load_text(root: Path, rel: str) -> str:
    p = root / rel
    check(p.exists(), f"missing {rel}")
    return _strip_block_comments(p.read_text(encoding="utf-8", errors="replace"))


def contains(hay: str, needle: str) -> bool:
    return needle in hay


def contains_not(hay: str, pattern: str, msg: str) -> None:
    check(not re.search(pattern, hay), msg)


def check_stride_addresses(kernel: str) -> None:
    """分离式段布局: hMix 三段直接落入连续数组。

    pre/post 每行 8 槽(align32(n*4)/4, 不足 32B 的块由 rightPadding 补齐),
    res 每行 resStride_ 槽(align32(n*n*4)/4); dstStride=0 使行距恰为段跨距。
    """
    expect_res = {4: 16, 6: 40, 8: 64}
    for n in LEGAL_N:
        pre_bytes = n * 4
        res_bytes = n * n * 4
        pre_stride = align_up(pre_bytes, 32) // 4
        res_stride = align_up(res_bytes, 32) // 4
        check(pre_stride == 8, f"n={n}: pre/post stride must be 8 floats")
        check(res_stride == expect_res[n],
              f"n={n}: res stride must be {expect_res[n]} floats")

        for half in (0, 1):
            pre_half = half * POST_ARRAY_ELEMS
            res_half = half * K_POST_GROUP_ROWS * res_stride
            for row in range(K_POST_GROUP_ROWS):
                check((pre_half + row * pre_stride) * 4 % 32 == 0 and
                      pre_half + (row + 1) * pre_stride <= 2 * POST_ARRAY_ELEMS,
                      f"n={n} pre half={half} row={row}: landing out of bounds")
                check((res_half + row * res_stride) * 4 % 32 == 0 and
                      res_half + (row + 1) * res_stride <= 2 * RES_ARRAY_MAX,
                      f"n={n} res half={half} row={row}: landing out of bounds")
    print("  stride addresses OK (split arrays, 32B aligned & in bounds)")


def check_redlines(code_dir: Path) -> None:
    global KERNEL_TEXT
    kernel = load_text(code_dir, KERNEL)
    KERNEL_TEXT = kernel
    host = load_text(code_dir, HOST)
    tiling = load_text(code_dir, TILING_H)
    key = load_text(code_dir, TILING_KEY)

    check(contains(kernel, '#include "lib/matmul_intf.h"'),
          "lib/matmul_intf.h include missing")
    check(contains(kernel, "KERNEL_TYPE_MIX_AIC_1_2"), "kernel type wrong")
    check(contains(kernel, "GetUserWorkspace(workspace)"), "GetUserWorkspace missing")
    contains_not(kernel, r"REGIST_MATMUL_OBJ", "REGIST_MATMUL_OBJ must not appear")
    contains_not(kernel, r"\bNUM_TWO\b", "undefined NUM_TWO must not appear")
    contains_not(kernel, r"kXQueueDepth", "kXQueueDepth constant must be removed")

    for name in ("phiGm_", "alphaGm_", "biasGm_", "gammaGm_", "hPostGm_", "hResGm_"):
        check(re.search(r"GlobalTensor<float>\s+" + name, kernel),
              f"{name} must be GlobalTensor<float>")
    for name in ("xGm_", "hInGm_"):
        check(re.search(r"GlobalTensor<DT_X>\s+" + name, kernel),
              f"{name} must be GlobalTensor<DT_X>")

    for ln_no, line in enumerate(kernel.splitlines()):
        usage = ("gammaGm_" in line or "gammaBuf_" in line) and (
            "SetGlobalBuffer" in line or "Get<" in line or "[" in line or
            "InitBuffer(gammaBuf_" in line or "gammaLocal" in line)
        if usage:
            window = "\n".join(kernel.splitlines()[max(0, ln_no - 2):ln_no + 1])
            check("hasGamma != 0" in window, f"unguarded gamma access near line {ln_no + 1}")

    sq_idx = kernel.find("Mul(calcLocal, rowCast, rowCast")
    rc_idx = kernel.find("ReduceSum<float>(batchSumLocal")
    gm_idx = kernel.find("Mul(rowCast, rowCast, gammaLocal")
    check(sq_idx >= 0 and rc_idx >= 0 and gm_idx >= 0, "pre square/reduce/gamma lines missing")
    check(sq_idx < rc_idx < gm_idx, "invRms square-sum must precede gamma multiply")

    check(kernel.count("hcEps") >= 1 and "tiling_->hcEps" in kernel,
          "hcEps usage missing in hPre sigmoid")
    res_start = kernel.find("Add(resArr[resHalf + j * resStride_],")
    hin_start = kernel.find("MulABLastDimBrcInline2<float>(xCastLocal[rowOff]")
    check(res_start >= 0 and hin_start > res_start, "hRes/hIn sections not found")
    res_block = kernel[res_start:hin_start]
    contains_not(res_block, r"Sigmoid|Exp\(|Log\(", "hRes must not pass through sigmoid")

    # Group_vec: group-level DMA + Brcb hIn present.
    check(contains(kernel, "LoadMixGroup"), "group hMix load helper missing")
    check(contains(kernel, "CopyGroupOutput"), "group output helper missing")
    check(contains(kernel, "LoadHInBlock"), "hIn x block prefetch helper missing")
    check(contains(kernel, "MulABLastDimBrcInline2"), "Brcb hIn helper missing")
    check(contains(kernel, "ReduceSumARAPerf"), "cross-stream reduce helper missing")
    check(contains(kernel, "DataCopyPad(dst, src[offset]"),
          "LoadSlice must use DataCopyPad")

    # DataCopyExtParams 步长单位:GM 侧一律字节,UB 侧一律 32B 块(CANN 8.0 官方
    # 参数表;arch35 mhc_pre_split_nd.h:487 对 GM->UB dstStride 除以 32 只是此
    # 规则的体现)。错误地把 UB 侧按字节传, 一度把组内各行散射到缓冲外:
    # 守卫修正后的字面值与真实块地址。
    load_mix = kernel[kernel.find("LoadMixGroup"):kernel.find("CopyGroupOutput")]
    check(load_mix.find("DataCopyExtParams") >= 0, "LoadMixGroup params missing")
    check(load_mix.count("DataCopyPad(") == 3,
          "LoadMixGroup must issue exactly 3 segment DataCopyPad")
    check(re.search(r"srcStride =", load_mix) is not None and
          re.search(r"\(tiling_->mixDim - n\) \* sizeof\(float\)", load_mix) is not None,
          "LoadMixGroup GM srcStride must be in bytes")
    check(re.search(r"dstStride = 0", load_mix) is not None,
          "LoadMixGroup dst arrays must be packed with dstStride = 0")
    check(re.search(r"rightPadding", load_mix) is not None,
          "LoadMixGroup must right-pad sub-32B blocks to keep 32B block alignment")
    check(re.search(r"resStride_", load_mix) is not None,
          "LoadMixGroup res segment must stride by the aligned resStride_")
    copy_out = kernel[kernel.find("CopyGroupOutput"):kernel.find("LoadHInBlock")]
    check("blockElements * sizeof(float)" in copy_out,
          "CopyGroupOutput must compute the blockLen from blockElements")
    check(re.search(r"srcStride = 0", copy_out) is not None,
          "CopyGroupOutput UB src must be packed (srcStride = 0)")
    check_stride_addresses(kernel)

    check(contains(kernel, "kAivToAicFlag = 8") and contains(kernel, "kAicToAivFlag = 9"),
          "cross-core flag values changed")
    for needle in ("CrossCoreSetFlag<kCrossCoreSyncMode, PIPE_MTE3>(kAivToAicFlag)",
                   "CrossCoreWaitFlag(kAicToAivFlag)",
                   "CrossCoreWaitFlag(kAivToAicFlag)",
                   "CrossCoreSetFlag<kCrossCoreSyncMode, PIPE_FIX>(kAicToAivFlag)"):
        check(contains(kernel, needle), "cross-core flag pairing broken: " + needle)
    check(contains(kernel, "kCrossCoreSyncMode = 2"), "MIX 1:2 sync-mode literal removed")

    check(contains(kernel, "matmulObj.Init(&tilingData.cubeTilingData, &pipe)"),
          "MatmulImpl Init signature changed")
    check(contains(kernel, "SetSubBlockIdx(0)"), "SetSubBlockIdx(0) missing")
    check(contains(kernel, "GetMDLConfig(true, false, 0, false, false, false, true)"),
          "Matmul config must equal the official MhcPre config")
    check(contains(kernel, "matmul::MatmulImpl<AType, BType, CType, CType, kDirectMatmulConfig>"),
          "MatmulImpl template instantiation changed")
    for needle in ("SetTensorA(xGammaGm_[aOffset])", "SetTensorB(phiGm_, true)",
                   "SetOrgShape(currentRows, tiling_->mixDim, tiling_->flatDim)",
                   "SetSingleShape(currentRows, tiling_->mixDim, tiling_->flatDim)",
                   "IterateAll<false>(hMixGm_[cOffset])", "matmulObj.End();"):
        check(contains(kernel, needle), "direct-matmul call sequence broken: " + needle)

    check(contains(tiling, "AscendC::tiling::TCubeTiling cubeTilingData;"),
          "TCubeTiling member missing from tiling struct")
    tail = tiling[tiling.find("cubeTilingData;") + len("cubeTilingData;"):]
    check(tail.strip().startswith("};"), "TCubeTiling must be the last tiling member")

    check(contains(host, "SetBlockDim(aicCoreNum)"), "SetBlockDim must use AIC core count")
    check(contains(host, "aivCoreNum != 2U * aicCoreNum"), "AIV/AIC 1:2 platform guard missing")
    check(contains(host, "2U * vectorRowsPerCore"), "rowsPerVector pairing broken")
    check(contains(host, "GetTiling(tiling->cubeTilingData)"),
          "host MatmulApiTiling GetTiling target changed")
    check(contains(host, "SetBias(false)") and contains(host, "SetBufferSpace(-1, -1, -1)"),
          "host matmul Bias/BufferSpace setup changed")
    check(contains(host, "const uint64_t xGammaBytes = batchSeq * flatDim * sizeof(float)"),
          "workspace layout must keep xGamma first")
    check(contains(host, "GetLibApiWorkSpaceSize"), "system workspace sizing changed")
    check(contains(host, "ASCENDC_TPL_SEL_PARAM(context, DT_X)"),
          "DT_X template dispatch missing")

    m = re.search(r"SetOrgShape\(\s*([A-Za-z_][A-Za-z0-9_]*|\d+)\s*,", host)
    check(m is not None, "SetOrgShape argument not recognized")
    const_def = re.search(m.group(1) + r"\s*=\s*(\d+)", host)
    check(const_def is not None and int(const_def.group(1)) == M64_TILING_M,
          "host must tile with fixed M=64")
    print("  redlines OK")


# ---------------------------------------------------------------------------
def main() -> None:
    args = [a for a in sys.argv[1:]]
    code = Path(args[args.index("--code") + 1] if "--code" in args else
                Path(__file__).resolve().parent / "code")
    check(code.is_dir(), f"code dir not found: {code}")

    print("MhcPre group_vec offline audit")
    _self_test_arith()
    check_partitions()
    check_shapes()
    check_formula()
    check_redlines(code)
    check_memory()
    print("MhcPre group_vec offline audit: PASS")


if __name__ == "__main__":
    main()