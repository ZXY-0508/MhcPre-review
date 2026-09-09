/**
 * E1/E2 probe for the group_vec MhcPre candidate.
 *
 * E1: MIX_AIC_1_2 8/9 cross-core protocol smoke on A2 - one minimal launch
 *     must complete (no hang) and produce correct hIn/hPost/hRes.
 * E2: the SAME aclOpExecutor is launched a second time (kernel-side this is
 *     the same MatmulImpl object hitting IterateAll again) - outputs of the
 *     second launch must still match the CPU reference.
 *
 * Build:  ./build_runner.sh (produces ./runner) then
 *         g++ -std=c++17 -O2 -o probe probe_e1e2.cpp <same flags>
 * Or simply: bash run_all.sh
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include "acl/acl.h"
#include "aclnn_mhc_pre.h"

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

static uint16_t FloatToFp16(float v)
{
    uint32_t bits;
    memcpy(&bits, &v, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp16 = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    if (exp16 >= 0x1F) {
        return (uint16_t)(sign | 0x7C00);
    }
    if (exp16 <= 0) {
        uint32_t m = mant | 0x800000;
        int shift = 14 - exp16;
        if (shift >= 25) {
            return (uint16_t)sign;
        }
        uint32_t half = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1);
        if (rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (half & 1))) {
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
    return (uint16_t)(sign | ((uint32_t)exp16 << 10) | mant16);
}

static float RintHalfEven(float v)
{
    float fl = std::floor(v);
    float frac = v - fl;
    if (frac > 0.5f || (frac == 0.5f && (int64_t)fl % 2 == 1)) {
        return fl + 1.0f;
    }
    return fl;
}

static void RunCpuReference(int64_t B, int64_t S, int64_t n, int64_t D,
                            const std::vector<uint16_t> &xRaw, const std::vector<float> &phi,
                            const std::vector<float> &alpha, const std::vector<float> &bias,
                            const std::vector<float> &gamma,
                            std::vector<float> &hIn_ref, std::vector<float> &hPost_ref,
                            std::vector<float> &hRes_ref)
{
    const int64_t BS = B * S;
    const int64_t flat = n * D;
    const int64_t mix = n * n + 2 * n;
    std::vector<float> xf(BS * flat);
    for (int64_t i = 0; i < BS * flat; ++i) {
        xf[i] = Fp16ToFloat(xRaw[i]);
    }
    std::vector<float> invRms(BS);
    for (int64_t r = 0; r < BS; ++r) {
        double sum = 0.0;
        for (int64_t k = 0; k < flat; ++k) {
            double v = xf[r * flat + k];
            sum += v * v;
        }
        invRms[r] = (float)(1.0 / std::sqrt(sum / flat + 1e-6f));
    }
    std::vector<float> hMix(BS * mix);
    for (int64_t r = 0; r < BS; ++r) {
        for (int64_t j = 0; j < mix; ++j) {
            float acc = 0.0f;
            for (int64_t k = 0; k < flat; ++k) {
                acc += xf[r * flat + k] * gamma[k] * phi[j * flat + k];
            }
            hMix[r * mix + j] = acc * invRms[r];
        }
    }
    hIn_ref.assign(BS * D, 0.0f);
    hPost_ref.assign(BS * n, 0.0f);
    hRes_ref.assign(BS * n * n, 0.0f);
    std::vector<float> hPre(BS * n);
    for (int64_t r = 0; r < BS; ++r) {
        for (int64_t j = 0; j < n; ++j) {
            float wPre = hMix[r * mix + j] * alpha[0] + bias[j];
            hPre[r * n + j] = 1.0f / (1.0f + std::exp(-wPre)) + 1e-6f;
            float wPost = hMix[r * mix + n + j] * alpha[1] + bias[n + j];
            hPost_ref[r * n + j] = 2.0f / (1.0f + std::exp(-wPost));
        }
        for (int64_t j = 0; j < n * n; ++j) {
            hRes_ref[r * n * n + j] = hMix[r * mix + 2 * n + j] * alpha[2] + bias[2 * n + j];
        }
        for (int64_t dd = 0; dd < D; ++dd) {
            float acc = 0.0f;
            for (int64_t s = 0; s < n; ++s) {
                acc += hPre[r * n + s] * xf[r * flat + s * D + dd];
            }
            uint16_t q = FloatToFp16(RintHalfEven(acc));
            hIn_ref[r * D + dd] = Fp16ToFloat(q);
        }
    }
}

static int Check(const std::vector<float> &ref, void *dev, int64_t elems, int64_t dtypeBytes,
                 const char *name)
{
    std::vector<uint8_t> raw(elems * dtypeBytes);
    if (aclrtMemcpy(raw.data(), elems * dtypeBytes, dev, elems * dtypeBytes,
                    ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        fprintf(stderr, "  [%s] copy out failed\n", name);
        return 1;
    }
    int64_t bad = 0;
    for (int64_t i = 0; i < elems; ++i) {
        float got;
        if (dtypeBytes == 2) {
            uint16_t v;
            memcpy(&v, raw.data() + i * 2, 2);
            got = Fp16ToFloat(v);
        } else {
            memcpy(&got, raw.data() + i * 4, 4);
        }
        double denom = std::max({1.0, (double)std::fabs(ref[i]), (double)std::fabs(got)});
        if (std::fabs(ref[i] - got) / denom > 5e-2) {
            bad++;
        }
    }
    printf("  [%s] %lld/%lld matched\n", name, (long long)(elems - bad), (long long)elems);
    return bad > elems / 1000 ? 1 : 0;
}

static int CreateRaw(const std::vector<uint16_t> &raw, const std::vector<int64_t> &shape,
                     void *&addr, aclTensor *&tensor)
{
    int64_t bytes = (int64_t)raw.size() * 2;
    if (aclrtMalloc(&addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemcpy(addr, bytes, raw.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return -1;
    }
    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT16, nullptr, 0, ACL_FORMAT_ND,
                             shape.data(), shape.size(), addr);
    return tensor ? 0 : -1;
}

static int CreateF32(const std::vector<float> &host, const std::vector<int64_t> &shape,
                     void *&addr, aclTensor *&tensor)
{
    int64_t bytes = (int64_t)host.size() * 4;
    if (aclrtMalloc(&addr, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemcpy(addr, bytes, host.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        return -1;
    }
    tensor = aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND,
                             shape.data(), shape.size(), addr);
    return tensor ? 0 : -1;
}

static int CreateOut(const std::vector<int64_t> &shape, aclDataType dt, void *&addr,
                     aclTensor *&tensor)
{
    int64_t elems = 1;
    for (int64_t d : shape) {
        elems *= d;
    }
    if (aclrtMalloc(&addr, elems * (dt == ACL_FLOAT ? 4 : 2), ACL_MEM_MALLOC_HUGE_FIRST) !=
        ACL_SUCCESS) {
        return -1;
    }
    tensor = aclCreateTensor(shape.data(), shape.size(), dt, nullptr, 0, ACL_FORMAT_ND,
                             shape.data(), shape.size(), addr);
    return tensor ? 0 : -1;
}

int main()
{
    if (aclInit(nullptr) != ACL_SUCCESS) {
        fprintf(stderr, "aclInit failed\n");
        return 1;
    }
    aclrtContext ctx = nullptr;
    aclrtStream stream = nullptr;
    if (aclrtSetDevice(0) != ACL_SUCCESS || aclrtCreateContext(&ctx, 0) != ACL_SUCCESS ||
        aclrtSetCurrentContext(ctx) != ACL_SUCCESS || aclrtCreateStream(&stream) != ACL_SUCCESS) {
        fprintf(stderr, "acl runtime init failed\n");
        return 1;
    }

    // Minimal shape: exercises the 8/9 handshake plus grouped postprocess tails.
    const int64_t B = 1, S = 64, n = 4, D = 256;
    const int64_t BS = B * S, flat = n * D, mix = n * n + 2 * n;
    printf("== E1/E2 probe: B=%lld S=%lld n=%lld D=%lld fp16 gamma=on ==\n",
           (long long)B, (long long)S, (long long)n, (long long)D);

    std::vector<int64_t> xShape = {B, S, n, D};
    std::vector<int64_t> phiShape = {flat, mix};
    std::vector<int64_t> alphaShape = {3};
    std::vector<int64_t> biasShape = {mix};
    std::vector<int64_t> gammaShape = {n, D};
    std::vector<int64_t> hInShape = {B, S, D};
    std::vector<int64_t> hPostShape = {B, S, n};
    std::vector<int64_t> hResShape = {B, S, n, n};

    std::vector<float> xHost(BS * flat), phiHost(flat * mix), alphaHost = {0.5f, 0.3f, 0.2f},
                             biasHost(mix), gammaHost(flat);
    srand(7);
    for (auto &v : xHost) {
        v = (float)(rand() % 2000) / 10000.0f - 0.1f;
    }
    for (auto &v : phiHost) {
        v = (float)(rand() % 200) / 10000.0f - 0.01f;
    }
    for (auto &v : biasHost) {
        v = (float)(rand() % 200) / 10000.0f - 0.01f;
    }
    for (auto &v : gammaHost) {
        v = 1.0f + (float)(rand() % 200) / 10000.0f;
    }
    std::vector<uint16_t> xRaw(BS * flat);
    for (size_t i = 0; i < xRaw.size(); ++i) {
        xRaw[i] = FloatToFp16(xHost[i]);
    }
    std::vector<float> hInRef, hPostRef, hResRef;
    RunCpuReference(B, S, n, D, xRaw, phiHost, alphaHost, biasHost, gammaHost, hInRef, hPostRef,
                    hResRef);

    void *xAddr = nullptr, *phiAddr = nullptr, *alphaAddr = nullptr, *biasAddr = nullptr,
         *gammaAddr = nullptr, *hInAddr = nullptr, *hPostAddr = nullptr, *hResAddr = nullptr,
         *workspace = nullptr;
    aclTensor *x = nullptr, *phi = nullptr, *alpha = nullptr, *bias = nullptr, *gamma = nullptr,
              *hIn = nullptr, *hPost = nullptr, *hRes = nullptr;
    int rc = 0;
    rc |= CreateRaw(xRaw, xShape, xAddr, x);
    rc |= CreateF32(phiHost, phiShape, phiAddr, phi);
    rc |= CreateF32(alphaHost, alphaShape, alphaAddr, alpha);
    rc |= CreateF32(biasHost, biasShape, biasAddr, bias);
    rc |= CreateF32(gammaHost, gammaShape, gammaAddr, gamma);
    rc |= CreateOut(hInShape, ACL_FLOAT16, hInAddr, hIn);
    rc |= CreateOut(hPostShape, ACL_FLOAT, hPostAddr, hPost);
    rc |= CreateOut(hResShape, ACL_FLOAT, hResAddr, hRes);
    if (rc != 0) {
        fprintf(stderr, "tensor creation failed\n");
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    if (aclnnMhcPreGetWorkspaceSize(x, phi, alpha, bias, gamma, 1e-6f, 1e-6f, hIn, hPost, hRes,
                                    &workspaceSize, &executor) != ACL_SUCCESS) {
        fprintf(stderr, "GetWorkspaceSize failed\n");
        return 1;
    }
    if (workspaceSize > 0 &&
        aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
        fprintf(stderr, "workspace malloc failed\n");
        return 1;
    }

    // ---- E1: first launch must complete (hang here = protocol broken) ----
    printf("E1: first launch (8/9 handshake + first IterateAll)...\n");
    if (aclnnMhcPre(workspace, workspaceSize, executor, stream) != ACL_SUCCESS ||
        aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        fprintf(stderr, "E1 FAIL: launch/sync failed\n");
        return 1;
    }
    printf("E1: launch completed\n");
    rc |= Check(hInRef, hInAddr, BS * D, 2, "E1 hIn");
    rc |= Check(hPostRef, hPostAddr, BS * n, 4, "E1 hPost");
    rc |= Check(hResRef, hResAddr, BS * n * n, 4, "E1 hRes");
    if (rc != 0) {
        fprintf(stderr, "E1 FAIL: outputs wrong\n");
        return 1;
    }
    printf("E1 PASS\n");

    // ---- E2: same executor, second launch (same MatmulImpl, IterateAll again) ----
    printf("E2: second launch on the same executor...\n");
    if (aclnnMhcPre(workspace, workspaceSize, executor, stream) != ACL_SUCCESS ||
        aclrtSynchronizeStream(stream) != ACL_SUCCESS) {
        fprintf(stderr, "E2 FAIL: relaunch failed\n");
        return 1;
    }
    rc |= Check(hInRef, hInAddr, BS * D, 2, "E2 hIn");
    rc |= Check(hPostRef, hPostAddr, BS * n, 4, "E2 hPost");
    rc |= Check(hResRef, hResAddr, BS * n * n, 4, "E2 hRes");
    if (rc != 0) {
        fprintf(stderr, "E2 FAIL: outputs wrong on relaunch\n");
        return 1;
    }
    printf("E2 PASS\n");

    aclDestroyTensor(x);
    aclDestroyTensor(phi);
    aclDestroyTensor(alpha);
    aclDestroyTensor(bias);
    aclDestroyTensor(gamma);
    aclDestroyTensor(hIn);
    aclDestroyTensor(hPost);
    aclDestroyTensor(hRes);
    aclrtFree(xAddr);
    aclrtFree(phiAddr);
    aclrtFree(alphaAddr);
    aclrtFree(biasAddr);
    aclrtFree(gammaAddr);
    aclrtFree(hInAddr);
    aclrtFree(hPostAddr);
    aclrtFree(hResAddr);
    if (workspace) {
        aclrtFree(workspace);
    }
    aclrtDestroyStream(stream);
    aclrtDestroyContext(ctx);
    aclrtResetDevice(0);
    aclFinalize();
    printf("E1/E2 probe: ALL PASS\n");
    return 0;
}
