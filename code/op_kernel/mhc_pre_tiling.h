// Copyright (c) 2026.
// Runtime tiling shared by the host and Ascend C kernel.
//
// 评测环境编译 kernel 时使用平台自带的原始 mhc_pre_tiling.h(提交包内该头文件
// 不生效)。本文件保持与平台原始结构逐字段一致, host 写入与 kernel 读取才能
// 对上布局; 字段名/顺序/宽度不可改动。
#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct MhcPreTilingData {
    uint64_t batchSeq;
    uint64_t streamCount;
    uint64_t headDim;
    uint64_t flatDim;
    uint64_t mixDim;
    uint64_t rowsPerVector;      // Matmul 侧的 M 切分(由 tiler 决定)
    uint64_t vectorRowsPerCore;  // 向量侧的行切分, 恒满足 aivCoreNum*该值 >= batchSeq
    uint32_t aicCoreNum;
    uint32_t aivCoreNum;
    uint32_t tileElements;
    uint32_t hasGamma;
    float invFlatDim;
    float normEps;
    float hcEps;
    AscendC::tiling::TCubeTiling cubeTilingData;
};
