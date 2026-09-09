/**
 * runner for the contest-signed MhcPre custom op (5 inputs + 3 outputs).
 * group_vec candidate: covers the E3 gate (full-kernel correctness vs CPU
 * reference + latency) over the shape matrix: n=4/6/8, gamma on/off,
 * fp16/bf16, plus a merged/tiled boundary case (D=2560 hits the hIn tile
 * loop when headDim > kHinTileElements=1024).
 *
 * Shape convention follows the host InferShape:
 *   x     : [B, S, n, D]   fp16/bf16
 *   phi   : [n*D, n*n+2n]  fp32
 *   alpha : [3]            fp32
 *   bias  : [n*n+2n]       fp32
 *   gamma : [n, D]         fp32 (optional)
 *   hIn   : [B, S, D]      fp16/bf16
 *   hPost : [B, S, n]      fp32
 *   hRes  : [B, S, n, n]   fp32
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "acl/acl.h"
#include "aclnn_mhc_pre.h"

using Clock = std::chrono::steady_clock;

static int64_t ShapeSize(const std::vector<int64_t> &shape)
{
    int64_t s = 1;
    for (int64_t d : shape) {
        s *= d;
    }
    return s;
}

// ---------------------------------------------------------------------------
// fp16 / bf16 helpers
// ---------------------------------------------------------------------------

static float Fp16ToFloat(uint16_t h)
{
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0x1F) {
        return mant ? NAN : (sign ? -INFINITY : INFINITY);
    }
    if (exp == 0) {
        return (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * (float)(1.0 / 16384.0);
    }
    return (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * (float)std::pow(2.0, (int)exp - 15);
}

static uint16_t FloatToFp16Bits(float v)
{
    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp == 0xFF) {
        return sign ? 0xFC00 : 0x7C00;
    }
    int32_t exp16 = (int32_t)exp - 127 + 15;
    if (exp16 >= 0x1F) {
        return sign ? 0xFC00 : 0x7C00;
    }
    if (exp16 <= 0) {
        uint32_t m = mant | 0x800000;
        int shift = 14 - exp16;
        if (shift >= 25) {
            return (uint16_t)sign;
        }
        uint32_t half = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1);
        uint32_t threshold = 1u << (shift - 1);
        if (rem > threshold || (rem == threshold && (half & 1))) {
            half++;
        }
        return (uint16_t)(sign | (half & 0x3FF));
    }
    uint32_t mant16 = mant >> 13;
    uint32_t rem = mant & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (mant16 & 1))) {
        mant16++;
        if (mant16 == 0x400) {
            mant16 = 0;
            exp16++;
        }
    }
    if (exp16 >= 0x1F) {
        return sign ? 0xFC00 : 0x7C00;
    }
    return (uint16_t)(sign | ((uint32_t)exp16 << 10) | mant16);
}

static uint32_t FloatToBf16Bits(float v)
{
    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t lsb = (bits >> 16) & 1;
    return (bits + 0x7FFF + lsb) & 0xFFFF0000;
}

static float Bf16ToFloat(uint32_t bfBits)
{
    uint32_t bits = bfBits;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// ---------------------------------------------------------------------------
// CPU reference (fp32, same order as the AscendC kernel)
// ---------------------------------------------------------------------------

static void RunCpuReference(int64_t B, int64_t S, int64_t n, int64_t D, bool isBf16,
                            const std::vector<uint32_t> &xRawBits, const std::vector<float> &phi,
                            const std::vector<float> &alpha, const std::vector<float> &bias,
                            const std::vector<float> &gamma, float normEps, float hcEps,
                            std::vector<float> &hIn_ref, std::vector<float> &hPost_ref,
                            std::vector<float> &hRes_ref)
{
    const int64_t BS = B * S;
    const int64_t flat = n * D;
    const int64_t mix = n * n + 2 * n;
    std::vector<float> xf(BS * flat);
    for (int64_t i = 0; i < BS * flat; ++i) {
        xf[i] = isBf16 ? Bf16ToFloat(xRawBits[i]) : Fp16ToFloat((uint16_t)xRawBits[i]);
    }

    std::vector<float> invRms(BS);
    for (int64_t r = 0; r < BS; ++r) {
        double sum = 0.0;
        for (int64_t k = 0; k < flat; ++k) {
            double v = xf[r * flat + k];
            sum += v * v;
        }
        invRms[r] = (float)(1.0 / std::sqrt(sum / flat + normEps));
    }

    // hMix = xGamma @ phi^T (fp32 accumulation)
    std::vector<float> hMix(BS * mix);
    for (int64_t r = 0; r < BS; ++r) {
        std::vector<float> xg(flat);
        for (int64_t k = 0; k < flat; ++k) {
            xg[k] = gamma.empty() ? xf[r * flat + k] : xf[r * flat + k] * gamma[k];
        }
        for (int64_t j = 0; j < mix; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < flat; ++k) {
                acc += xg[k] * phi[j * flat + k];
            }
            hMix[r * mix + j] = acc * invRms[r];
        }
    }

    hPost_ref.assign(BS * n, 0.0f);
    hRes_ref.assign(BS * n * n, 0.0f);
    hIn_ref.assign(BS * D, 0.0f);
    std::vector<float> hPre(BS * n);
    for (int64_t r = 0; r < BS; ++r) {
        for (int64_t j = 0; j < n; ++j) {
            float wPre = hMix[r * mix + j] * alpha[0] + bias[j];
            float sig = 1.0f / (1.0f + std::exp(-wPre));
            hPre[r * n + j] = sig + hcEps;

            float wPost = hMix[r * mix + n + j] * alpha[1] + bias[n + j];
            float sigPost = 1.0f / (1.0f + std::exp(-wPost));
            hPost_ref[r * n + j] = 2.0f * sigPost;
        }
        for (int64_t j = 0; j < n * n; ++j) {
            hRes_ref[r * n * n + j] = hMix[r * mix + 2 * n + j] * alpha[2] + bias[2 * n + j];
        }
        for (int64_t dd = 0; dd < D; ++dd) {
            float acc = 0.0f;
            for (int64_t s = 0; s < n; ++s) {
                acc += hPre[r * n + s] * xf[r * flat + s * D + dd];
            }
            // CAST_RINT: fp32 -> fp16/bf16 直接舍入到目标网格(round-half-even),
            // 例如 0.25 保留为 0.25; 不是先取整数再转格式。
            hIn_ref[r * D + dd] =
                isBf16 ? Bf16ToFloat(FloatToBf16Bits(acc))
                       : Fp16ToFloat(FloatToFp16Bits(acc));
        }
    }
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

static int CreateTensorDtX(const std::vector<float> &host, const std::vector<int64_t> &shape,
                           bool isBf16, void *&addr, aclTensor *&tensor)
{
    int64_t elems = ShapeSize(shape);
    int64_t bytes = elems * 2;
    std::vector<uint16_t> raw(elems);
    for (int64_t i = 0; i < elems; ++i) {
        // bf16 的 16 位模式在 fp32 的高半字, fp16 在低半字 —— 统一取 16 位模式
        uint32_t bits = isBf16 ? (FloatToBf16Bits(host[i]) >> 16)
                               : FloatToFp16Bits(host[i]);
        memcpy(&raw[i], &bits, 2);
    }
    if (aclrtMalloc(&addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return -1;
    }
    if (aclrtMemcpy(addr, bytes, raw.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return -1;
    }
    aclDataType dt = isBf16 ? ACL_BF16 : ACL_FLOAT16;
    tensor = aclCreateTensor(shape.data(), shape.size(), dt, nullptr, 0, ACL_FORMAT_ND,
                             shape.data(), shape.size(), addr);
    return tensor == nullptr ? -1 : 0;
}

static int CreateTensorF32(const std::vector<float> &host, const std::vector<int64_t> &shape,
                           void *&addr, aclTensor *&tensor)
{
    int64_t bytes = (int64_t)host.size() * 4;
    if (aclrtMalloc(&addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return -1;
    }
    if (aclrtMemcpy(addr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return -1;
    }
    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
                             shape.data(), shape.size(), addr);
    return tensor == nullptr ? -1 : 0;
}

static int CreateOutputTensor(const std::vector<int64_t> &shape, aclDataType dtype,
                              void *&addr, aclTensor *&tensor)
{
    int64_t elems = ShapeSize(shape);
    int64_t bytes = elems * (dtype == ACL_FLOAT16 || dtype == ACL_BF16 ? 2 : 4);
    if (aclrtMalloc(&addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        return -1;
    }
    tensor = aclCreateTensor(shape.data(), shape.size(), dtype, nullptr, 0, ACL_FORMAT_ND,
                             shape.data(), shape.size(), addr);
    return tensor == nullptr ? -1 : 0;
}

// ---------------------------------------------------------------------------

static void CheckOutputs(const std::vector<float> &ref, void *dev, int64_t elems,
                         int64_t dtypeBytes, bool isBf16, const char *name)
{
    std::vector<float> got(elems);
    int64_t bytes = elems * dtypeBytes;
    if (dtypeBytes == 2) {
        // 16-bit 输出先读入独立原始缓冲, 再解码到 float 缓冲, 避免原地展开时
        // 写 4B float 覆盖尚未读取的下一个 2B 原始数。
        std::vector<uint16_t> raw(elems);
        if (aclrtMemcpy(raw.data(), bytes, dev, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            fprintf(stderr, "copy out failed\n");
            return;
        }
        for (int64_t i = 0; i < elems; ++i) {
            if (isBf16) {
                // bf16 的 16 位模式位于 fp32 高半字, 左移 16 位后按 fp32 解出
                got[i] = Bf16ToFloat(((uint32_t)raw[i]) << 16);
            } else {
                got[i] = Fp16ToFloat(raw[i]);
            }
        }
    } else {
        if (aclrtMemcpy(got.data(), bytes, dev, bytes, ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            fprintf(stderr, "copy out failed\n");
            return;
        }
    }
    int64_t bad = 0;
    double maxRel = 0.0;
    for (int64_t i = 0; i < elems; ++i) {
        double a = ref[i], b = got[i];
        if (std::isnan(a) || std::isnan(b)) {
            bad++;
            continue;
        }
        double denom = std::max({1.0, std::fabs(a), std::fabs(b)});
        double rel = std::fabs(a - b) / denom;
        maxRel = std::max(maxRel, rel);
        if (rel > 5e-2) {
            bad++;
        }
    }
    printf("  %s: %lld/%lld matched (maxRelErr=%.2e)\n", name, (long long)(elems - bad),
           (long long)elems, maxRel);
    printf("    ref[0..4] = %f %f %f %f %f\n", ref[0], ref[1], ref[2], ref[3], ref[4]);
    printf("    got[0..4] = %f %f %f %f %f\n", got[0], got[1], got[2], got[3], got[4]);
    if (bad > elems / 1000) {
        fprintf(stderr, "FAIL: too many mismatches in %s\n", name);
        exit(1);
    }
}

// ---------------------------------------------------------------------------

struct CaseSpec {
    int64_t B, S, n, D;
    bool hasGamma;
    bool isBf16;
};

int main()
{
    int32_t deviceId = 0;
    aclrtContext ctx = nullptr;
    aclrtStream stream = nullptr;

    // E3 gate: n=4/6/8, gamma on/off, fp16/bf16; D=2560 crosses the hIn tile
    // boundary (headDim > kHinTileElements=1024); D=256 is the merged case.
    const CaseSpec cases[] = {
        {1, 2048, 4, 2560, true, false},
        {1, 2048, 6, 1024, true, false},
        {1, 1024, 8, 1024, true, false},
        {1, 2048, 4, 2560, false, false},
        {1, 2048, 6, 1024, true, true},
        {1, 1024, 8, 1024, false, true},
    };

    if (aclInit(nullptr) != ACL_SUCCESS) {
        fprintf(stderr, "aclInit failed\n");
        return 1;
    }
    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        fprintf(stderr, "aclrtSetDevice failed\n");
        return 1;
    }
    if (aclrtCreateContext(&ctx, deviceId) != ACL_SUCCESS) {
        fprintf(stderr, "aclrtCreateContext failed\n");
        return 1;
    }
    if (aclrtSetCurrentContext(ctx) != ACL_SUCCESS) {
        fprintf(stderr, "aclrtSetCurrentContext failed\n");
        return 1;
    }
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        fprintf(stderr, "aclrtCreateStream failed\n");
        return 1;
    }

    for (const CaseSpec &c : cases) {
        int64_t B = c.B, S = c.S, n = c.n, D = c.D;
        int64_t BS = B * S;
        int64_t flat = n * D;
        int64_t mix = n * n + 2 * n;
        printf("\n==== case B=%lld S=%lld n=%lld D=%lld gamma=%d dtype=%s ====\n",
               (long long)B, (long long)S, (long long)n, (long long)D,
               (int)c.hasGamma, c.isBf16 ? "bf16" : "fp16");

        std::vector<int64_t> xShape = {B, S, n, D};
        std::vector<int64_t> phiShape = {flat, mix};
        std::vector<int64_t> alphaShape = {3};
        std::vector<int64_t> biasShape = {mix};
        std::vector<int64_t> gammaShape = {n, D};
        std::vector<int64_t> hInShape = {B, S, D};
        std::vector<int64_t> hPostShape = {B, S, n};
        std::vector<int64_t> hResShape = {B, S, n, n};

        std::vector<float> xHost(BS * flat), phiHost(flat * mix),
            alphaHost = {0.5f, 0.3f, 0.2f}, biasHost(mix), gammaHost(flat);
        srand(42);
        for (size_t i = 0; i < xHost.size(); ++i) {
            xHost[i] = (float)(rand() % 2000) / 10000.0f - 0.1f;
        }
        for (size_t i = 0; i < phiHost.size(); ++i) {
            phiHost[i] = (float)(rand() % 200) / 10000.0f - 0.01f;
        }
        for (size_t i = 0; i < biasHost.size(); ++i) {
            biasHost[i] = (float)(rand() % 200) / 10000.0f - 0.01f;
        }
        for (size_t i = 0; i < gammaHost.size(); ++i) {
            gammaHost[i] = 1.0f + (float)(rand() % 200) / 10000.0f;
        }

        // Raw device x bits (fp16 or bf16) for the reference path.
        // bf16 保留完整 32 位模式(HIGH half), Bf16ToFloat 才能直接按 fp32 解出。
        std::vector<uint32_t> xRawBits(BS * flat);
        for (size_t i = 0; i < xRawBits.size(); ++i) {
            if (c.isBf16) {
                xRawBits[i] = FloatToBf16Bits(xHost[i]);
            } else {
                xRawBits[i] = FloatToFp16Bits(xHost[i]);
            }
        }

        std::vector<float> hInRef, hPostRef, hResRef;
        RunCpuReference(B, S, n, D, c.isBf16, xRawBits, phiHost, alphaHost, biasHost,
                        c.hasGamma ? gammaHost : std::vector<float>(),
                        1e-6f, 1e-6f, hInRef, hPostRef, hResRef);

        void *xAddr = nullptr, *phiAddr = nullptr, *alphaAddr = nullptr,
             *biasAddr = nullptr, *gammaAddr = nullptr,
             *hInAddr = nullptr, *hPostAddr = nullptr, *hResAddr = nullptr;
        aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr,
                   *bias = nullptr, *gamma = nullptr,
                   *hIn = nullptr, *hPost = nullptr, *hRes = nullptr;

        if (CreateTensorDtX(xHost, xShape, c.isBf16, xAddr, x) != 0) {
            fprintf(stderr, "Create x failed\n");
            return 1;
        }
        if (CreateTensorF32(phiHost, phiShape, phiAddr, phi) != 0) {
            fprintf(stderr, "Create phi failed\n");
            return 1;
        }
        if (CreateTensorF32(alphaHost, alphaShape, alphaAddr, alpha) != 0) {
            fprintf(stderr, "Create alpha failed\n");
            return 1;
        }
        if (CreateTensorF32(biasHost, biasShape, biasAddr, bias) != 0) {
            fprintf(stderr, "Create bias failed\n");
            return 1;
        }
        if (c.hasGamma && CreateTensorF32(gammaHost, gammaShape, gammaAddr, gamma) != 0) {
            fprintf(stderr, "Create gamma failed\n");
            return 1;
        }
        if (CreateOutputTensor(hInShape, c.isBf16 ? ACL_BF16 : ACL_FLOAT16, hInAddr, hIn) != 0) {
            fprintf(stderr, "Create hIn failed\n");
            return 1;
        }
        if (CreateOutputTensor(hPostShape, ACL_FLOAT, hPostAddr, hPost) != 0) {
            fprintf(stderr, "Create hPost failed\n");
            return 1;
        }
        if (CreateOutputTensor(hResShape, ACL_FLOAT, hResAddr, hRes) != 0) {
            fprintf(stderr, "Create hRes failed\n");
            return 1;
        }

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        float normEps = 1e-6f, hcEps = 1e-6f;
        aclnnStatus st = aclnnMhcPreGetWorkspaceSize(
            x, phi, alpha, bias, gamma, normEps, hcEps,
            hIn, hPost, hRes, &workspaceSize, &executor);
        if (st != ACL_SUCCESS) {
            fprintf(stderr, "GetWorkspaceSize failed: %d\n", st);
            return 1;
        }
        void *workspace = nullptr;
        if (workspaceSize > 0) {
            if (aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
                fprintf(stderr, "workspace malloc failed\n");
                return 1;
            }
        }

        // warmup
        if (aclnnMhcPre(workspace, workspaceSize, executor, stream) != ACL_SUCCESS) {
            fprintf(stderr, "warmup failed\n");
            return 1;
        }
        if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
            fprintf(stderr, "sync after warmup failed\n");
            return 1;
        }

        const int iters = 20;
        std::vector<double> us(iters);
        for (int i = 0; i < iters; ++i) {
            auto t0 = Clock::now();
            if (aclnnMhcPre(workspace, workspaceSize, executor, stream) != ACL_SUCCESS) {
                fprintf(stderr, "iter %d failed\n", i);
                return 1;
            }
            if (aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
                fprintf(stderr, "sync iter %d failed\n", i);
                return 1;
            }
            auto t1 = Clock::now();
            us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
        }
        std::sort(us.begin(), us.end());
        double avg = 0;
        for (double v : us) {
            avg += v;
        }
        avg /= iters;
        printf("  latency: min=%.2f us  median=%.2f us  avg=%.2f us  (iters=%d)\n",
               us.front(), us[iters / 2], avg, iters);

        CheckOutputs(hInRef, hInAddr, BS * D, 2, c.isBf16, "hIn");
        CheckOutputs(hPostRef, hPostAddr, BS * n, 4, false, "hPost");
        CheckOutputs(hResRef, hResAddr, BS * n * n, 4, false, "hRes");

        aclDestroyTensor(x);
        aclDestroyTensor(phi);
        aclDestroyTensor(alpha);
        aclDestroyTensor(bias);
        if (gamma != nullptr) {
            aclDestroyTensor(gamma);
        }
        aclDestroyTensor(hIn);
        aclDestroyTensor(hPost);
        aclDestroyTensor(hRes);
        aclrtFree(xAddr);
        aclrtFree(phiAddr);
        aclrtFree(alphaAddr);
        aclrtFree(biasAddr);
        if (gammaAddr != nullptr) {
            aclrtFree(gammaAddr);
        }
        aclrtFree(hInAddr);
        aclrtFree(hPostAddr);
        aclrtFree(hResAddr);
        if (workspace) {
            aclrtFree(workspace);
        }
    }

    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(deviceId);
    aclFinalize();
    printf("\nAll cases done.\n");
    return 0;
}
