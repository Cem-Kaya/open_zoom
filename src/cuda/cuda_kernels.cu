#ifdef _WIN32

#include "openzoom/cuda/cuda_kernels.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#if __has_include(<cuda_surface_types.h>)
#include <cuda_surface_types.h>
#elif __has_include(<surface_types.h>)
#include <surface_types.h>
#else
#error "cuda surface type header not found"
#endif
#if __has_include(<surface_functions.h>)
#include <surface_functions.h>
#endif
#include <math.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

namespace openzoom {

namespace {

__device__ unsigned char FloatToByte(float value) {
    const float clamped = fminf(fmaxf(value, 0.0f), 1.0f);
    return static_cast<unsigned char>(clamped * 255.0f);
}

__device__ float3 ReadPixelLinear(const uchar4* src, size_t pitchBytes, int x, int y, int width, int height) {
    const int clampedX = max(0, min(width - 1, x));
    const int clampedY = max(0, min(height - 1, y));
    const auto* row = reinterpret_cast<const uchar4*>(reinterpret_cast<const uint8_t*>(src) + pitchBytes * clampedY);
    const uchar4 value = row[clampedX];
    return make_float3(value.x, value.y, value.z);
}

__device__ float Lanczos2(float x) {
    x = fabsf(x);
    if (x < 1e-6f) {
        return 1.0f;
    }
    if (x >= 2.0f) {
        return 0.0f;
    }
    const float pix = CUDART_PI_F * x;
    const float pixHalf = pix * 0.5f;
    return (sinf(pix) * sinf(pixHalf)) / (pix * pixHalf);
}

__device__ float3 LanczosSample(const uchar4* src, size_t pitchBytes,
                                float sampleX, float sampleY,
                                int width, int height) {
    const int baseX = static_cast<int>(floorf(sampleX));
    const int baseY = static_cast<int>(floorf(sampleY));
    float3 accum = make_float3(0.0f, 0.0f, 0.0f);
    float weightSum = 0.0f;
    for (int j = -1; j <= 2; ++j) {
        for (int i = -1; i <= 2; ++i) {
            const float w = Lanczos2(sampleX - (baseX + i)) * Lanczos2(sampleY - (baseY + j));
            const float3 c = ReadPixelLinear(src, pitchBytes, baseX + i, baseY + j, width, height);
            accum.x += c.x * w;
            accum.y += c.y * w;
            accum.z += c.z * w;
            weightSum += w;
        }
    }
    if (weightSum > 0.0f) {
        const float inv = 1.0f / weightSum;
        accum.x *= inv;
        accum.y *= inv;
        accum.z *= inv;
    }
    return accum;
}

__device__ float3 BilinearSample(const uchar4* src, size_t pitchBytes,
                                 float sampleX, float sampleY,
                                 int width, int height) {
    const float fx = floorf(sampleX);
    const float fy = floorf(sampleY);
    const int x0 = static_cast<int>(fx);
    const int y0 = static_cast<int>(fy);
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float tx = sampleX - fx;
    const float ty = sampleY - fy;
    const float3 c00 = ReadPixelLinear(src, pitchBytes, x0, y0, width, height);
    const float3 c10 = ReadPixelLinear(src, pitchBytes, x1, y0, width, height);
    const float3 c01 = ReadPixelLinear(src, pitchBytes, x0, y1, width, height);
    const float3 c11 = ReadPixelLinear(src, pitchBytes, x1, y1, width, height);

    const float3 c0 = make_float3(c00.x + tx * (c10.x - c00.x),
                                  c00.y + tx * (c10.y - c00.y),
                                  c00.z + tx * (c10.z - c00.z));
    const float3 c1 = make_float3(c01.x + tx * (c11.x - c01.x),
                                  c01.y + tx * (c11.y - c01.y),
                                  c01.z + tx * (c11.z - c01.z));

    return make_float3(c0.x + ty * (c1.x - c0.x),
                       c0.y + ty * (c1.y - c0.y),
                       c0.z + ty * (c1.z - c0.z));
}

__device__ float3 BoxBlur3x3(const uchar4* src, size_t pitchBytes,
                             int x, int y, int width, int height) {
    float3 accum = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            const float3 c = ReadPixelLinear(src, pitchBytes, x + i, y + j, width, height);
            accum.x += c.x;
            accum.y += c.y;
            accum.z += c.z;
        }
    }
    const float inv = 1.0f / 9.0f;
    accum.x *= inv;
    accum.y *= inv;
    accum.z *= inv;
    return accum;
}

__global__ void GradientFillKernel(cudaSurfaceObject_t surface, int width, int height, float timeSeconds) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const float fx = static_cast<float>(x) / static_cast<float>(width);
    const float fy = static_cast<float>(y) / static_cast<float>(height);
    const float phase = timeSeconds * 0.25f;

    const float r = 0.5f + 0.5f * sinf((fx + phase) * 6.28318f);
    const float g = 0.5f + 0.5f * sinf((fy + phase) * 6.28318f + 2.09439f);
    const float b = 0.5f + 0.5f * sinf(((fx + fy) * 0.5f + phase) * 6.28318f + 4.18878f);

    uchar4 color = make_uchar4(FloatToByte(b), FloatToByte(g), FloatToByte(r), 255);
    surf2Dwrite(color, surface, x * sizeof(uchar4), y);
}

} // namespace

void LaunchGradientKernel(cudaSurfaceObject_t surface, int width, int height, float timeSeconds) {
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);

    GradientFillKernel<<<gridSize, blockSize>>>(surface, width, height, timeSeconds);
}

namespace {

__device__ unsigned char ToGrayscaleChannel(const uchar4& pixel) {
    const float r = static_cast<float>(pixel.z) / 255.0f;
    const float g = static_cast<float>(pixel.y) / 255.0f;
    const float b = static_cast<float>(pixel.x) / 255.0f;
    const float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
    return FloatToByte(luminance);
}

__global__ void BlackWhiteKernel(cudaSurfaceObject_t surface, int width, int height, float threshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const uchar4 pixel = surf2Dread<uchar4>(surface, x * static_cast<int>(sizeof(uchar4)), y);
    const float thresholdClamped = fminf(fmaxf(threshold, 0.0f), 1.0f);
    const unsigned char luminance = ToGrayscaleChannel(pixel);
    const unsigned char value = (static_cast<float>(luminance) / 255.0f) >= thresholdClamped ? 255 : 0;
    const uchar4 bwPixel = make_uchar4(value, value, value, pixel.w);
    surf2Dwrite(bwPixel, surface, x * sizeof(uchar4), y);
}

__global__ void ZoomKernel(cudaSurfaceObject_t surface, int width, int height, float zoomAmount) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const float zoom = fmaxf(zoomAmount, 1.0f);

    const float nx = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
    const float ny = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);

    const float cx = 0.5f;
    const float cy = 0.5f;

    const float srcX = (nx - cx) / zoom + cx;
    const float srcY = (ny - cy) / zoom + cy;

    if (srcX < 0.0f || srcX > 1.0f || srcY < 0.0f || srcY > 1.0f) {
        surf2Dwrite(make_uchar4(0, 0, 0, 255), surface, x * sizeof(uchar4), y);
        return;
    }

    const int sampleX = static_cast<int>(srcX * static_cast<float>(width));
    const int sampleY = static_cast<int>(srcY * static_cast<float>(height));

    const int clampedX = max(0, min(width - 1, sampleX));
    const int clampedY = max(0, min(height - 1, sampleY));

    const uchar4 sampled = surf2Dread<uchar4>(surface, clampedX * static_cast<int>(sizeof(uchar4)), clampedY);
    surf2Dwrite(sampled, surface, x * sizeof(uchar4), y);
}

} // namespace

void LaunchBlackWhiteKernel(cudaSurfaceObject_t surface, int width, int height, float threshold) {
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);

    BlackWhiteKernel<<<gridSize, blockSize>>>(surface, width, height, threshold);
}

void LaunchZoomKernel(cudaSurfaceObject_t surface, int width, int height, float zoomAmount) {
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);

    ZoomKernel<<<gridSize, blockSize>>>(surface, width, height, zoomAmount);
}

namespace {

__device__ inline const uchar4* RowAt(const uchar4* base, size_t pitchBytes, int y) {
    return reinterpret_cast<const uchar4*>(reinterpret_cast<const char*>(base) + static_cast<size_t>(y) * pitchBytes);
}

__device__ inline uchar4* RowAt(uchar4* base, size_t pitchBytes, int y) {
    return reinterpret_cast<uchar4*>(reinterpret_cast<char*>(base) + static_cast<size_t>(y) * pitchBytes);
}

__device__ inline float4* RowAt(float4* base, size_t pitchBytes, int y) {
    return reinterpret_cast<float4*>(reinterpret_cast<char*>(base) + static_cast<size_t>(y) * pitchBytes);
}

__global__ void BlackWhiteLinearKernel(uchar4* dst, size_t dstPitch,
                                       const uchar4* src, size_t srcPitch,
                                       int width, int height, float threshold) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const uchar4* srcRow = RowAt(src, srcPitch, y);
    uchar4 pixel = srcRow[x];
    const float thresholdClamped = fminf(fmaxf(threshold, 0.0f), 1.0f);
    const unsigned char luminance = ToGrayscaleChannel(pixel);
    const unsigned char value = (static_cast<float>(luminance) / 255.0f) >= thresholdClamped ? 255 : 0;
    pixel.x = value;
    pixel.y = value;
    pixel.z = value;

    uchar4* dstRow = RowAt(dst, dstPitch, y);
    dstRow[x] = pixel;
}

__global__ void ZoomLinearKernel(uchar4* dst, size_t dstPitch,
                                 const uchar4* src, size_t srcPitch,
                                 int width, int height,
                                 float zoomAmount, float centerXNorm, float centerYNorm) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float zoom = fmaxf(zoomAmount, 1.0f);
    const float maxIndexX = static_cast<float>(max(width - 1, 0));
    const float maxIndexY = static_cast<float>(max(height - 1, 0));

    float centerX = fminf(fmaxf(centerXNorm, 0.0f), 1.0f) * maxIndexX;
    float centerY = fminf(fmaxf(centerYNorm, 0.0f), 1.0f) * maxIndexY;

    const float halfVisibleWidth = (static_cast<float>(width)) / (zoom * 2.0f);
    const float halfVisibleHeight = (static_cast<float>(height)) / (zoom * 2.0f);

    if (width > 1) {
        const float minCenterX = fmaxf(0.0f, halfVisibleWidth - 0.5f);
        const float maxCenterX = fminf(maxIndexX, static_cast<float>(width) - 1.0f - (halfVisibleWidth - 0.5f));
        if (minCenterX <= maxCenterX) {
            centerX = fminf(fmaxf(centerX, minCenterX), maxCenterX);
        }
    }

    if (height > 1) {
        const float minCenterY = fmaxf(0.0f, halfVisibleHeight - 0.5f);
        const float maxCenterY = fminf(maxIndexY, static_cast<float>(height) - 1.0f - (halfVisibleHeight - 0.5f));
        if (minCenterY <= maxCenterY) {
            centerY = fminf(fmaxf(centerY, minCenterY), maxCenterY);
        }
    }

    const float outputCenterX = maxIndexX * 0.5f;
    const float outputCenterY = maxIndexY * 0.5f;

    const float sx = (static_cast<float>(x) - outputCenterX) / zoom + centerX;
    const float sy = (static_cast<float>(y) - outputCenterY) / zoom + centerY;

    int sampleX = static_cast<int>(roundf(sx));
    int sampleY = static_cast<int>(roundf(sy));
    sampleX = max(0, min(width - 1, sampleX));
    sampleY = max(0, min(height - 1, sampleY));

    const uchar4* srcRow = RowAt(src, srcPitch, sampleY);
    const uchar4 sampled = srcRow[sampleX];

    uchar4* dstRow = RowAt(dst, dstPitch, y);
    dstRow[x] = sampled;
}

constexpr int kMaxBlurRadius = 50;
__constant__ float gGaussianKernel[(kMaxBlurRadius * 2) + 1];
__constant__ int gGaussianRadius;
__constant__ int gGaussianKernelSize;

__global__ void GaussianBlurHorizontalKernel(uchar4* dst, size_t dstPitch,
                                             const uchar4* src, size_t srcPitch,
                                             int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int radius = gGaussianRadius;

    float accumB = 0.0f;
    float accumG = 0.0f;
    float accumR = 0.0f;
    float accumA = 0.0f;

    const uchar4* srcRow = RowAt(src, srcPitch, y);
    for (int k = -radius; k <= radius; ++k) {
        int sampleX = x + k;
        sampleX = max(0, min(width - 1, sampleX));
        const uchar4 sample = srcRow[sampleX];
        const float weight = gGaussianKernel[k + radius];
        accumB += weight * sample.x;
        accumG += weight * sample.y;
        accumR += weight * sample.z;
        accumA += weight * sample.w;
    }

    uchar4* dstRow = RowAt(dst, dstPitch, y);
    dstRow[x] = make_uchar4(FloatToByte(accumB / 255.0f),
                             FloatToByte(accumG / 255.0f),
                             FloatToByte(accumR / 255.0f),
                             FloatToByte(accumA / 255.0f));
}

__global__ void GaussianBlurVerticalKernel(uchar4* dst, size_t dstPitch,
                                           const uchar4* src, size_t srcPitch,
                                           int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int radius = gGaussianRadius;

    float accumB = 0.0f;
    float accumG = 0.0f;
    float accumR = 0.0f;
    float accumA = 0.0f;

    for (int k = -radius; k <= radius; ++k) {
        int sampleY = y + k;
        sampleY = max(0, min(height - 1, sampleY));
        const uchar4* srcRow = RowAt(src, srcPitch, sampleY);
        const uchar4 sample = srcRow[x];
        const float weight = gGaussianKernel[k + radius];
        accumB += weight * sample.x;
        accumG += weight * sample.y;
        accumR += weight * sample.z;
        accumA += weight * sample.w;
    }

    uchar4* dstRow = RowAt(dst, dstPitch, y);
    dstRow[x] = make_uchar4(FloatToByte(accumB / 255.0f),
                             FloatToByte(accumG / 255.0f),
                             FloatToByte(accumR / 255.0f),
                             FloatToByte(accumA / 255.0f));
}

__global__ void FocusMarkerKernel(uchar4* buffer, size_t pitchBytes,
                                  int width, int height,
                                  float centerXNorm, float centerYNorm,
                                  float outerRadius, float innerRadius) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float centerX = centerXNorm * (width - 1.0f);
    const float centerY = centerYNorm * (height - 1.0f);
    const float dx = static_cast<float>(x) - centerX;
    const float dy = static_cast<float>(y) - centerY;
    const float distSq = dx * dx + dy * dy;

    if (distSq > outerRadius * outerRadius) {
        return;
    }

    uchar4* row = RowAt(buffer, pitchBytes, y);

    if (distSq <= innerRadius * innerRadius) {
        row[x] = make_uchar4(255, 255, 255, 255);
    } else {
        row[x] = make_uchar4(0, 0, 255, 255);
    }
}

__global__ void TemporalSmoothKernel(uchar4* dst, size_t dstPitch,
                                     const uchar4* src, size_t srcPitch,
                                     float4* history, size_t historyPitch,
                                     int width, int height,
                                     float alpha, int historySeed) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float clampedAlpha = fminf(fmaxf(alpha, 0.0f), 1.0f);
    const float oneMinusAlpha = 1.0f - clampedAlpha;
    const bool hasHistory = historySeed != 0;

    const uchar4* srcRow = RowAt(src, srcPitch, y);
    uchar4 current = srcRow[x];

    float4* historyRow = RowAt(history, historyPitch, y);
    float4 previous = hasHistory ? historyRow[x]
                                 : make_float4(static_cast<float>(current.x),
                                              static_cast<float>(current.y),
                                              static_cast<float>(current.z),
                                              255.0f);

    const float currB = static_cast<float>(current.x);
    const float currG = static_cast<float>(current.y);
    const float currR = static_cast<float>(current.z);

    float4 blended;
    blended.x = clampedAlpha * currB + oneMinusAlpha * previous.x;
    blended.y = clampedAlpha * currG + oneMinusAlpha * previous.y;
    blended.z = clampedAlpha * currR + oneMinusAlpha * previous.z;
    blended.w = 255.0f;

    blended.x = fminf(fmaxf(blended.x, 0.0f), 255.0f);
    blended.y = fminf(fmaxf(blended.y, 0.0f), 255.0f);
    blended.z = fminf(fmaxf(blended.z, 0.0f), 255.0f);

    historyRow[x] = blended;

    uchar4* dstRow = RowAt(dst, dstPitch, y);
    dstRow[x] = make_uchar4(static_cast<unsigned char>(blended.x + 0.5f),
                            static_cast<unsigned char>(blended.y + 0.5f),
                            static_cast<unsigned char>(blended.z + 0.5f),
                            255u);
}

__global__ void FsrEasuRcasKernel(uchar4* dst, size_t dstPitchBytes,
                                  const uchar4* src, size_t srcPitchBytes,
                                  int srcWidth, int srcHeight,
                                  int dstWidth, int dstHeight,
                                  float sharpness) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    const float scaleX = static_cast<float>(srcWidth) / static_cast<float>(dstWidth);
    const float scaleY = static_cast<float>(srcHeight) / static_cast<float>(dstHeight);
    const float sampleX = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
    const float sampleY = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;

    float3 color = LanczosSample(src, srcPitchBytes, sampleX, sampleY, srcWidth, srcHeight);

    const int nearX = static_cast<int>(roundf(sampleX));
    const int nearY = static_cast<int>(roundf(sampleY));
    const float3 blur = BoxBlur3x3(src, srcPitchBytes, nearX, nearY, srcWidth, srcHeight);

    const float sharpen = fmaxf(0.0f, fminf(sharpness, 1.0f));
    color.x = fmaf(sharpen, color.x - blur.x, color.x);
    color.y = fmaf(sharpen, color.y - blur.y, color.y);
    color.z = fmaf(sharpen, color.z - blur.z, color.z);

    color.x = fminf(fmaxf(color.x, 0.0f), 255.0f);
    color.y = fminf(fmaxf(color.y, 0.0f), 255.0f);
    color.z = fminf(fmaxf(color.z, 0.0f), 255.0f);

    auto* row = RowAt(dst, dstPitchBytes, y);
    row[x] = make_uchar4(static_cast<unsigned char>(color.x + 0.5f),
                         static_cast<unsigned char>(color.y + 0.5f),
                         static_cast<unsigned char>(color.z + 0.5f),
                         255u);
}

__global__ void NisKernel(uchar4* dst, size_t dstPitchBytes,
                          const uchar4* src, size_t srcPitchBytes,
                          int srcWidth, int srcHeight,
                          int dstWidth, int dstHeight,
                          float sharpness) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstWidth || y >= dstHeight) {
        return;
    }

    const float scaleX = static_cast<float>(srcWidth) / static_cast<float>(dstWidth);
    const float scaleY = static_cast<float>(srcHeight) / static_cast<float>(dstHeight);
    const float sampleX = (static_cast<float>(x) + 0.5f) * scaleX - 0.5f;
    const float sampleY = (static_cast<float>(y) + 0.5f) * scaleY - 0.5f;

    float3 base = BilinearSample(src, srcPitchBytes, sampleX, sampleY, srcWidth, srcHeight);

    const int nearX = static_cast<int>(roundf(sampleX));
    const int nearY = static_cast<int>(roundf(sampleY));
    const float3 blur = BoxBlur3x3(src, srcPitchBytes, nearX, nearY, srcWidth, srcHeight);

    const float sharpen = fmaxf(0.0f, fminf(sharpness, 1.0f));
    base.x = fmaf(sharpen, base.x - blur.x, base.x);
    base.y = fmaf(sharpen, base.y - blur.y, base.y);
    base.z = fmaf(sharpen, base.z - blur.z, base.z);

    base.x = fminf(fmaxf(base.x, 0.0f), 255.0f);
    base.y = fminf(fmaxf(base.y, 0.0f), 255.0f);
    base.z = fminf(fmaxf(base.z, 0.0f), 255.0f);

    auto* row = RowAt(dst, dstPitchBytes, y);
    row[x] = make_uchar4(static_cast<unsigned char>(base.x + 0.5f),
                         static_cast<unsigned char>(base.y + 0.5f),
                         static_cast<unsigned char>(base.z + 0.5f),
                         255u);
}

inline void CheckCuda(const char* message) {
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

void LaunchBlackWhiteLinear(uchar4* dst, size_t dstPitchBytes,
                            const uchar4* src, size_t srcPitchBytes,
                            int width, int height, float threshold, cudaStream_t stream) {
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);
    BlackWhiteLinearKernel<<<gridSize, blockSize, 0, stream>>>(dst, dstPitchBytes, src, srcPitchBytes,
                                                              width, height, threshold);
    CheckCuda("BlackWhiteLinearKernel launch failed");
}

void LaunchZoomLinear(uchar4* dst, size_t dstPitchBytes,
                      const uchar4* src, size_t srcPitchBytes,
                      int width, int height,
                      float zoomAmount, float centerXNorm, float centerYNorm,
                      cudaStream_t stream) {
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);
    ZoomLinearKernel<<<gridSize, blockSize, 0, stream>>>(dst, dstPitchBytes, src, srcPitchBytes,
                                                        width, height, zoomAmount, centerXNorm, centerYNorm);
    CheckCuda("ZoomLinearKernel launch failed");
}

void LaunchGaussianBlurLinear(uchar4* dst, size_t dstPitchBytes,
                              uchar4* scratch, size_t scratchPitchBytes,
                              const uchar4* src, size_t srcPitchBytes,
                              int width, int height,
                              cudaStream_t stream) {
    if (width <= 0 || height <= 0) {
        return;
    }
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);

    GaussianBlurHorizontalKernel<<<gridSize, blockSize, 0, stream>>>(scratch, scratchPitchBytes, src, srcPitchBytes, width, height);
    CheckCuda("GaussianBlurHorizontalKernel launch failed");

    GaussianBlurVerticalKernel<<<gridSize, blockSize, 0, stream>>>(dst, dstPitchBytes, scratch, scratchPitchBytes, width, height);
    CheckCuda("GaussianBlurVerticalKernel launch failed");
}

void LaunchFocusMarkerLinear(uchar4* buffer, size_t pitchBytes,
                             int width, int height,
                             float centerXNorm, float centerYNorm,
                             cudaStream_t stream) {
    const float dimension = static_cast<float>(std::min(width, height));
    const float radiusOuter = fmaxf(6.0f, dimension * 0.03f);
    const float radiusInner = radiusOuter * 0.35f;

    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);

    FocusMarkerKernel<<<gridSize, blockSize, 0, stream>>>(buffer, pitchBytes, width, height,
                                                          centerXNorm, centerYNorm,
                                                          radiusOuter, radiusInner);
    CheckCuda("FocusMarkerKernel launch failed");
}

void LaunchFsrEasuRcasLinear(uchar4* dst, size_t dstPitchBytes,
                             const uchar4* src, size_t srcPitchBytes,
                             int srcWidth, int srcHeight,
                             int dstWidth, int dstHeight,
                             float sharpness,
                             cudaStream_t stream) {
    if (dstWidth <= 0 || dstHeight <= 0 || srcWidth <= 0 || srcHeight <= 0) {
        return;
    }
    const dim3 blockSize(16, 16);
    const dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x,
                        (dstHeight + blockSize.y - 1) / blockSize.y);
    FsrEasuRcasKernel<<<gridSize, blockSize, 0, stream>>>(dst, dstPitchBytes,
                                                         src, srcPitchBytes,
                                                         srcWidth, srcHeight,
                                                         dstWidth, dstHeight,
                                                         sharpness);
    CheckCuda("FsrEasuRcasKernel launch failed");
}

void LaunchNisLinear(uchar4* dst, size_t dstPitchBytes,
                     const uchar4* src, size_t srcPitchBytes,
                     int srcWidth, int srcHeight,
                     int dstWidth, int dstHeight,
                     float sharpness,
                     cudaStream_t stream) {
    if (dstWidth <= 0 || dstHeight <= 0 || srcWidth <= 0 || srcHeight <= 0) {
        return;
    }
    const dim3 blockSize(16, 16);
    const dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x,
                        (dstHeight + blockSize.y - 1) / blockSize.y);
    NisKernel<<<gridSize, blockSize, 0, stream>>>(dst, dstPitchBytes,
                                                 src, srcPitchBytes,
                                                 srcWidth, srcHeight,
                                                 dstWidth, dstHeight,
                                                 sharpness);
    CheckCuda("NisKernel launch failed");
}

void LaunchTemporalSmoothLinear(uchar4* dst, size_t dstPitchBytes,
                                const uchar4* src, size_t srcPitchBytes,
                                float4* history, size_t historyPitchBytes,
                                int width, int height,
                                float alpha,
                                bool historyValid,
                                cudaStream_t stream) {
    if (width <= 0 || height <= 0) {
        return;
    }

    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);

    TemporalSmoothKernel<<<gridSize, blockSize, 0, stream>>>(dst, dstPitchBytes,
                                                            src, srcPitchBytes,
                                                            history, historyPitchBytes,
                                                            width, height,
                                                            alpha,
                                                            historyValid ? 1 : 0);
    CheckCuda("TemporalSmoothKernel launch failed");
}

namespace {

inline void CheckCudaStatus(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

constexpr int kStabSearchRadius = 16;
constexpr int kStabCandidateCount = kStabSearchRadius * 2 + 1;
constexpr int kStabMinOverlap = 8;
constexpr int kStabEstimateBlockSize = 128;
constexpr float kStabInvalidSad = 3.402823466e+38f;

// Box-average a factorX x factorY block of the BGRA source into one luma value.
// Buffers are BGRA (uchar4: x=B, y=G, z=R, w=A), so luma = 0.299*z + 0.587*y + 0.114*x.
__global__ void StabilizationLumaDownsampleKernel(float* dstLuma,
                                                  int smallWidth, int smallHeight,
                                                  const uchar4* src, size_t srcPitch,
                                                  int width, int height,
                                                  int factorX, int factorY) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= smallWidth || y >= smallHeight) {
        return;
    }

    const int x0 = x * factorX;
    const int y0 = y * factorY;
    const int x1 = min(x0 + factorX, width);
    const int y1 = min(y0 + factorY, height);

    float sum = 0.0f;
    int count = 0;
    for (int sy = y0; sy < y1; ++sy) {
        const uchar4* row = RowAt(src, srcPitch, sy);
        for (int sx = x0; sx < x1; ++sx) {
            const uchar4 pixel = row[sx];
            sum += 0.299f * static_cast<float>(pixel.z) +
                   0.587f * static_cast<float>(pixel.y) +
                   0.114f * static_cast<float>(pixel.x);
            ++count;
        }
    }

    dstLuma[y * smallWidth + x] = (count > 0) ? sum / static_cast<float>(count) : 0.0f;
}

// Profiles are tiny (<= 320 floats), so atomicAdd contention is negligible.
__global__ void StabilizationProjectionKernel(const float* luma,
                                              int smallWidth, int smallHeight,
                                              float* colProj, float* rowProj) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= smallWidth || y >= smallHeight) {
        return;
    }

    const float value = luma[y * smallWidth + x];
    atomicAdd(&colProj[x], value);
    atomicAdd(&rowProj[y], value);
}

// SAD between the current profile and the previous profile displaced by `shift`
// (content moved by `shift`: curr[i] ~ prev[i - shift]), normalized by overlap.
__device__ float ProfileSad(const float* curr, const float* prev, int length, int shift) {
    float sum = 0.0f;
    int count = 0;
    for (int i = 0; i < length; ++i) {
        const int j = i - shift;
        if (j >= 0 && j < length) {
            sum += fabsf(curr[i] - prev[j]);
            ++count;
        }
    }
    return (count >= kStabMinOverlap) ? sum / static_cast<float>(count) : kStabInvalidSad;
}

// Single-block kernel: estimates per-frame translation from 1D projection
// profiles, integrates the camera path, low-passes it, and stores the clamped
// correction in device memory. Nothing is ever read back to the host.
__global__ void StabilizationEstimateKernel(const float* currColProj, const float* currRowProj,
                                            float* prevColProj, float* prevRowProj,
                                            int smallWidth, int smallHeight,
                                            float factorX, float factorY,
                                            int fullWidth, int fullHeight,
                                            float strength, int previousValid,
                                            StabilizationState* state) {
    __shared__ float sadX[kStabCandidateCount];
    __shared__ float sadY[kStabCandidateCount];

    const int tid = static_cast<int>(threadIdx.x);

    for (int c = tid; c < kStabCandidateCount; c += blockDim.x) {
        const int shift = c - kStabSearchRadius;
        sadX[c] = (previousValid != 0) ? ProfileSad(currColProj, prevColProj, smallWidth, shift)
                                       : kStabInvalidSad;
        sadY[c] = (previousValid != 0) ? ProfileSad(currRowProj, prevRowProj, smallHeight, shift)
                                       : kStabInvalidSad;
    }
    __syncthreads();

    if (tid == 0) {
        if (previousValid == 0) {
            // First frame after a reset: no motion reference yet.
            state->actualPath = make_float2(0.0f, 0.0f);
            state->smoothPath = make_float2(0.0f, 0.0f);
            state->correction = make_float2(0.0f, 0.0f);
        } else {
            // Argmin over candidate shifts; initializing with the zero-shift SAD
            // biases ties toward "no motion".
            int dx = 0;
            float bestX = sadX[kStabSearchRadius];
            int dy = 0;
            float bestY = sadY[kStabSearchRadius];
            for (int c = 0; c < kStabCandidateCount; ++c) {
                if (sadX[c] < bestX) {
                    bestX = sadX[c];
                    dx = c - kStabSearchRadius;
                }
                if (sadY[c] < bestY) {
                    bestY = sadY[c];
                    dy = c - kStabSearchRadius;
                }
            }

            float2 actual = state->actualPath;
            float2 smooth = state->smoothPath;
            // Small-image shift -> full-res pixels.
            actual.x += static_cast<float>(dx) * factorX;
            actual.y += static_cast<float>(dy) * factorY;

            // smoothPath = lerp(actualPath, smoothPath, strength): exponential
            // low-pass that follows intentional pans but suppresses jitter.
            const float s = fminf(fmaxf(strength, 0.0f), 0.98f);
            smooth.x = actual.x + s * (smooth.x - actual.x);
            smooth.y = actual.y + s * (smooth.y - actual.y);

            const float marginX = 0.06f * static_cast<float>(fullWidth);
            const float marginY = 0.06f * static_cast<float>(fullHeight);
            float2 correction = make_float2(smooth.x - actual.x, smooth.y - actual.y);
            correction.x = fminf(fmaxf(correction.x, -marginX), marginX);
            correction.y = fminf(fmaxf(correction.y, -marginY), marginY);

            // Re-center the paths occasionally so float precision cannot degrade
            // over very long sessions; their difference (the correction) is kept.
            if (fabsf(actual.x) > 1.0e6f) {
                smooth.x -= actual.x;
                actual.x = 0.0f;
            }
            if (fabsf(actual.y) > 1.0e6f) {
                smooth.y -= actual.y;
                actual.y = 0.0f;
            }

            state->actualPath = actual;
            state->smoothPath = smooth;
            state->correction = correction;
        }
    }
    __syncthreads();

    // Stash the current projections as "previous" for the next frame.
    for (int i = tid; i < smallWidth; i += blockDim.x) {
        prevColProj[i] = currColProj[i];
    }
    for (int i = tid; i < smallHeight; i += blockDim.x) {
        prevRowProj[i] = currRowProj[i];
    }
}

// Shift the frame by `correction` (out(x) = in(x - correction)): content the
// camera dragged to `actualPath` is re-rendered as if it sat at `smoothPath`.
// The correction is read straight from device memory (no host round trip).
__global__ void StabilizationWarpKernel(uchar4* dst, size_t dstPitch,
                                        const uchar4* src, size_t srcPitch,
                                        int width, int height,
                                        const StabilizationState* state) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float2 correction = state->correction;
    const float sampleX = static_cast<float>(x) - correction.x;
    const float sampleY = static_cast<float>(y) - correction.y;
    const float3 color = BilinearSample(src, srcPitch, sampleX, sampleY, width, height);

    uchar4* dstRow = RowAt(dst, dstPitch, y);
    dstRow[x] = make_uchar4(static_cast<unsigned char>(fminf(color.x + 0.5f, 255.0f)),
                            static_cast<unsigned char>(fminf(color.y + 0.5f, 255.0f)),
                            static_cast<unsigned char>(fminf(color.z + 0.5f, 255.0f)),
                            255u);
}

__device__ float ApplyContrastBrightness(float normalized, float contrast, float brightness) {
    return fminf(fmaxf((normalized - 0.5f) * contrast + 0.5f + brightness, 0.0f), 1.0f);
}

// Auto-contrast remap: stretch [lo, hi] to [0, 1], blended with the original
// value by `strength` (0 = untouched, 1 = full stretch). The range floor
// (16/255) prevents blowups on nearly flat frames.
__device__ float ApplyAutoContrast(float value, float lo, float hi, float strength) {
    const float range = fmaxf(hi - lo, 16.0f / 255.0f);
    const float stretched = fminf(fmaxf((value - lo) / range, 0.0f), 1.0f);
    return value + strength * (stretched - value);
}

// Contrast/brightness plus display color mode. Operates in place: each thread
// reads and writes only its own pixel, so no ping-pong buffer swap is needed.
// When enabled, the auto-contrast level stretch runs first (levels are read
// straight from device memory — no host readback), then contrast/brightness.
// Modes (values 0..1, buffers are BGRA):
//   0 Normal:        pass-through after contrast/brightness.
//   1 Inverted:      each color channel -> 1 - v (alpha unchanged).
//   2 WhiteOnBlack:  grayscale, r=g=b=L (bright text stays bright on black).
//   3 YellowOnBlack: r=g=L, b=0 (luma tinted yellow).
//   4 BlackOnYellow: r=g=L, b=0.15*L — bright background maps to yellow paper,
//                    dark text stays black; the small blue floor keeps the
//                    yellow slightly soft instead of fully saturated.
__global__ void DisplayColorGradeKernel(uchar4* buffer, size_t pitchBytes,
                                        int width, int height,
                                        int colorMode, float contrast, float brightness,
                                        const float2* autoContrastLevels,
                                        float autoContrastStrength) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    uchar4* row = RowAt(buffer, pitchBytes, y);
    const uchar4 pixel = row[x];

    float b = static_cast<float>(pixel.x) / 255.0f;
    float g = static_cast<float>(pixel.y) / 255.0f;
    float r = static_cast<float>(pixel.z) / 255.0f;

    if (autoContrastLevels != nullptr) {
        // One coalesced broadcast read per thread; the L2 cache absorbs it.
        const float2 levels = *autoContrastLevels;
        const float strength = fminf(fmaxf(autoContrastStrength, 0.0f), 1.0f);
        b = ApplyAutoContrast(b, levels.x, levels.y, strength);
        g = ApplyAutoContrast(g, levels.x, levels.y, strength);
        r = ApplyAutoContrast(r, levels.x, levels.y, strength);
    }

    b = ApplyContrastBrightness(b, contrast, brightness);
    g = ApplyContrastBrightness(g, contrast, brightness);
    r = ApplyContrastBrightness(r, contrast, brightness);

    switch (colorMode) {
    case 1: {
        b = 1.0f - b;
        g = 1.0f - g;
        r = 1.0f - r;
        break;
    }
    case 2: {
        const float luma = 0.299f * r + 0.587f * g + 0.114f * b;
        b = luma;
        g = luma;
        r = luma;
        break;
    }
    case 3: {
        const float luma = 0.299f * r + 0.587f * g + 0.114f * b;
        b = 0.0f;
        g = luma;
        r = luma;
        break;
    }
    case 4: {
        const float luma = 0.299f * r + 0.587f * g + 0.114f * b;
        b = 0.15f * luma;
        g = luma;
        r = luma;
        break;
    }
    default:
        break;
    }

    row[x] = make_uchar4(FloatToByte(b), FloatToByte(g), FloatToByte(r), pixel.w);
}

__device__ inline unsigned char ClampIntToByte(int value) {
    return static_cast<unsigned char>(max(0, min(255, value)));
}

// Integer BT.601 limited-range YUV -> BGRA. The coefficients and rounding match
// ConvertNv12ToBgra in src/common/image_processing.cpp bit-for-bit so switching
// the conversion from CPU to GPU changes nothing visually.
__device__ inline uchar4 Bt601ToBgra(int yValue, int u, int v) {
    const int c = max(yValue - 16, 0);
    const int r = (298 * c + 409 * v + 128) >> 8;
    const int g = (298 * c - 100 * u - 208 * v + 128) >> 8;
    const int b = (298 * c + 516 * u + 128) >> 8;
    return make_uchar4(ClampIntToByte(b), ClampIntToByte(g), ClampIntToByte(r), 255u);
}

__global__ void Nv12ToBgraKernel(uchar4* dst, size_t dstPitch,
                                 const unsigned char* yPlane, size_t yPitch,
                                 const unsigned char* uvPlane, size_t uvPitch,
                                 int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const unsigned char* yRow = yPlane + static_cast<size_t>(y) * yPitch;
    const unsigned char* uvRow = uvPlane + static_cast<size_t>(y >> 1) * uvPitch;
    const int uvIndex = x & ~1;
    const int u = static_cast<int>(uvRow[uvIndex]) - 128;
    const int v = static_cast<int>(uvRow[uvIndex + 1]) - 128;

    uchar4* dstRow = RowAt(dst, dstPitch, y);
    dstRow[x] = Bt601ToBgra(static_cast<int>(yRow[x]), u, v);
}

// One thread per horizontal pixel pair (YUY2 stores Y0 U Y1 V per 2 pixels).
__global__ void Yuy2ToBgraKernel(uchar4* dst, size_t dstPitch,
                                 const unsigned char* src, size_t srcPitch,
                                 int width, int height) {
    const int pairX = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int x0 = pairX * 2;
    if (x0 >= width || y >= height) {
        return;
    }

    const unsigned char* srcPtr = src + static_cast<size_t>(y) * srcPitch +
                                  static_cast<size_t>(pairX) * 4u;
    const int y0 = srcPtr[0];
    const int u = static_cast<int>(srcPtr[1]) - 128;
    const int y1 = srcPtr[2];
    const int v = static_cast<int>(srcPtr[3]) - 128;

    uchar4* dstRow = RowAt(dst, dstPitch, y);
    dstRow[x0] = Bt601ToBgra(y0, u, v);
    if (x0 + 1 < width) {
        dstRow[x0 + 1] = Bt601ToBgra(y1, u, v);
    }
}

// Clockwise quarter-turn rotation, gathered from the destination side so writes
// stay coalesced. dstWidth/dstHeight are the post-rotation extents.
__global__ void RotateQuarterKernel(uchar4* dst, size_t dstPitch,
                                    const uchar4* src, size_t srcPitch,
                                    int srcWidth, int srcHeight,
                                    int dstWidth, int dstHeight,
                                    int turns) {
    const int ox = blockIdx.x * blockDim.x + threadIdx.x;
    const int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= dstWidth || oy >= dstHeight) {
        return;
    }

    int ix = ox;
    int iy = oy;
    switch (turns) {
    case 1:  // 90 CW: dst(ox,oy) = src(oy, srcHeight-1-ox)
        ix = oy;
        iy = srcHeight - 1 - ox;
        break;
    case 2:  // 180
        ix = srcWidth - 1 - ox;
        iy = srcHeight - 1 - oy;
        break;
    case 3:  // 270 CW: dst(ox,oy) = src(srcWidth-1-oy, ox)
        ix = srcWidth - 1 - oy;
        iy = ox;
        break;
    default:
        break;
    }

    uchar4* dstRow = RowAt(dst, dstPitch, oy);
    dstRow[ox] = RowAt(src, srcPitch, iy)[ix];
}

// Perspective warp: each output pixel is mapped through the 3x3 homography
// (output px -> source px, row major, passed by value as kernel arguments so no
// constant-memory synchronization is needed) and sampled bilinearly. Pixels
// mapping outside the source rect are painted black.
struct KeystoneHomography {
    float m[9];
};

__global__ void KeystoneWarpKernel(uchar4* dst, size_t dstPitch,
                                   const uchar4* src, size_t srcPitch,
                                   int width, int height,
                                   KeystoneHomography h) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float fx = static_cast<float>(x);
    const float fy = static_cast<float>(y);
    const float w = h.m[6] * fx + h.m[7] * fy + h.m[8];

    uchar4* dstRow = RowAt(dst, dstPitch, y);
    if (fabsf(w) < 1e-8f) {
        dstRow[x] = make_uchar4(0, 0, 0, 255u);
        return;
    }

    const float invW = 1.0f / w;
    const float sx = (h.m[0] * fx + h.m[1] * fy + h.m[2]) * invW;
    const float sy = (h.m[3] * fx + h.m[4] * fy + h.m[5]) * invW;

    if (sx < 0.0f || sy < 0.0f ||
        sx > static_cast<float>(width - 1) || sy > static_cast<float>(height - 1)) {
        dstRow[x] = make_uchar4(0, 0, 0, 255u);
        return;
    }

    const float3 color = BilinearSample(src, srcPitch, sx, sy, width, height);
    dstRow[x] = make_uchar4(static_cast<unsigned char>(fminf(color.x + 0.5f, 255.0f)),
                            static_cast<unsigned char>(fminf(color.y + 0.5f, 255.0f)),
                            static_cast<unsigned char>(fminf(color.z + 0.5f, 255.0f)),
                            255u);
}

// 256-bin luma histogram. Full-frame rather than the small stabilization luma:
// the small image is produced by box-averaging, which squeezes the extremes and
// would bias the percentile levels inward; a full-frame pass is a single read
// per pixel (about the cost of the downsample itself) and measures the true
// distribution. Shared-memory block histograms keep global atomics to
// 256 per block. Requires blockDim.x * blockDim.y == 256.
__global__ void AutoContrastHistogramKernel(unsigned int* histogram,
                                            const uchar4* src, size_t srcPitch,
                                            int width, int height) {
    __shared__ unsigned int blockHist[256];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    blockHist[tid] = 0;
    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        const uchar4 pixel = RowAt(src, srcPitch, y)[x];
        const float luma = 0.299f * static_cast<float>(pixel.z) +
                           0.587f * static_cast<float>(pixel.y) +
                           0.114f * static_cast<float>(pixel.x);
        const int bin = min(255, max(0, __float2int_rn(luma)));
        atomicAdd(&blockHist[bin], 1u);
    }
    __syncthreads();

    if (blockHist[tid] != 0) {
        atomicAdd(&histogram[tid], blockHist[tid]);
    }
}

// Single-thread scan over 256 bins (nanoseconds) that finds the 2nd/98th
// percentile levels and low-passes them in device memory; the grade kernel
// reads the result directly, so auto contrast never touches the host.
__global__ void AutoContrastAnalysisKernel(const unsigned int* histogram,
                                           int pixelCount,
                                           float2* levels,
                                           int levelsValid) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    const unsigned int total = static_cast<unsigned int>(max(pixelCount, 1));
    const unsigned int lowTarget = static_cast<unsigned int>(
        (static_cast<unsigned long long>(total) * 2ull) / 100ull);
    const unsigned int highTarget = static_cast<unsigned int>(
        (static_cast<unsigned long long>(total) * 98ull) / 100ull);

    int lo = 0;
    int hi = 255;
    unsigned int cumulative = 0;
    bool loFound = false;
    for (int bin = 0; bin < 256; ++bin) {
        cumulative += histogram[bin];
        if (!loFound && cumulative > lowTarget) {
            lo = bin;
            loFound = true;
        }
        if (cumulative >= highTarget) {
            hi = bin;
            break;
        }
    }
    if (hi <= lo) {
        hi = min(lo + 1, 255);
    }

    const float2 measured = make_float2(static_cast<float>(lo) / 255.0f,
                                        static_cast<float>(hi) / 255.0f);
    if (levelsValid != 0) {
        // Temporal smoothing (lerp 0.1) so slide transitions do not flash.
        const float2 previous = *levels;
        *levels = make_float2(previous.x + 0.1f * (measured.x - previous.x),
                              previous.y + 0.1f * (measured.y - previous.y));
    } else {
        *levels = measured;
    }
}

} // namespace

void LaunchStabilizationLumaDownsample(float* dstLuma,
                                       int smallWidth, int smallHeight,
                                       const uchar4* src, size_t srcPitchBytes,
                                       int width, int height,
                                       int factorX, int factorY,
                                       cudaStream_t stream) {
    if (smallWidth <= 0 || smallHeight <= 0 || width <= 0 || height <= 0 ||
        factorX <= 0 || factorY <= 0) {
        return;
    }
    const dim3 blockSize(16, 16);
    const dim3 gridSize((smallWidth + blockSize.x - 1) / blockSize.x,
                        (smallHeight + blockSize.y - 1) / blockSize.y);
    StabilizationLumaDownsampleKernel<<<gridSize, blockSize, 0, stream>>>(dstLuma,
                                                                          smallWidth, smallHeight,
                                                                          src, srcPitchBytes,
                                                                          width, height,
                                                                          factorX, factorY);
    CheckCuda("StabilizationLumaDownsampleKernel launch failed");
}

void LaunchStabilizationProjections(const float* luma,
                                    int smallWidth, int smallHeight,
                                    float* colProj, float* rowProj,
                                    cudaStream_t stream) {
    if (smallWidth <= 0 || smallHeight <= 0) {
        return;
    }
    CheckCudaStatus(cudaMemsetAsync(colProj, 0, sizeof(float) * static_cast<size_t>(smallWidth), stream),
                    "cudaMemsetAsync column projection failed");
    CheckCudaStatus(cudaMemsetAsync(rowProj, 0, sizeof(float) * static_cast<size_t>(smallHeight), stream),
                    "cudaMemsetAsync row projection failed");
    const dim3 blockSize(16, 16);
    const dim3 gridSize((smallWidth + blockSize.x - 1) / blockSize.x,
                        (smallHeight + blockSize.y - 1) / blockSize.y);
    StabilizationProjectionKernel<<<gridSize, blockSize, 0, stream>>>(luma,
                                                                      smallWidth, smallHeight,
                                                                      colProj, rowProj);
    CheckCuda("StabilizationProjectionKernel launch failed");
}

void LaunchStabilizationEstimate(const float* currColProj, const float* currRowProj,
                                 float* prevColProj, float* prevRowProj,
                                 int smallWidth, int smallHeight,
                                 float factorX, float factorY,
                                 int fullWidth, int fullHeight,
                                 float strength,
                                 bool previousValid,
                                 StabilizationState* state,
                                 cudaStream_t stream) {
    if (smallWidth <= 0 || smallHeight <= 0 || fullWidth <= 0 || fullHeight <= 0) {
        return;
    }
    StabilizationEstimateKernel<<<1, kStabEstimateBlockSize, 0, stream>>>(currColProj, currRowProj,
                                                                          prevColProj, prevRowProj,
                                                                          smallWidth, smallHeight,
                                                                          factorX, factorY,
                                                                          fullWidth, fullHeight,
                                                                          strength,
                                                                          previousValid ? 1 : 0,
                                                                          state);
    CheckCuda("StabilizationEstimateKernel launch failed");
}

void LaunchStabilizationWarp(uchar4* dst, size_t dstPitchBytes,
                             const uchar4* src, size_t srcPitchBytes,
                             int width, int height,
                             const StabilizationState* state,
                             cudaStream_t stream) {
    if (width <= 0 || height <= 0) {
        return;
    }
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);
    StabilizationWarpKernel<<<gridSize, blockSize, 0, stream>>>(dst, dstPitchBytes,
                                                                src, srcPitchBytes,
                                                                width, height,
                                                                state);
    CheckCuda("StabilizationWarpKernel launch failed");
}

void LaunchDisplayColorGradeLinear(uchar4* buffer, size_t pitchBytes,
                                   int width, int height,
                                   int colorMode, float contrast, float brightness,
                                   const float2* autoContrastLevels,
                                   float autoContrastStrength,
                                   cudaStream_t stream) {
    if (width <= 0 || height <= 0) {
        return;
    }
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);
    DisplayColorGradeKernel<<<gridSize, blockSize, 0, stream>>>(buffer, pitchBytes,
                                                                width, height,
                                                                colorMode, contrast, brightness,
                                                                autoContrastLevels,
                                                                autoContrastStrength);
    CheckCuda("DisplayColorGradeKernel launch failed");
}

void LaunchNv12ToBgraLinear(uchar4* dst, size_t dstPitchBytes,
                            const unsigned char* yPlane, size_t yPitchBytes,
                            const unsigned char* uvPlane, size_t uvPitchBytes,
                            int width, int height,
                            cudaStream_t stream) {
    if (width <= 0 || height <= 0) {
        return;
    }
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);
    Nv12ToBgraKernel<<<gridSize, blockSize, 0, stream>>>(dst, dstPitchBytes,
                                                         yPlane, yPitchBytes,
                                                         uvPlane, uvPitchBytes,
                                                         width, height);
    CheckCuda("Nv12ToBgraKernel launch failed");
}

void LaunchYuy2ToBgraLinear(uchar4* dst, size_t dstPitchBytes,
                            const unsigned char* src, size_t srcPitchBytes,
                            int width, int height,
                            cudaStream_t stream) {
    if (width <= 0 || height <= 0) {
        return;
    }
    const int pairCount = (width + 1) / 2;
    const dim3 blockSize(16, 16);
    const dim3 gridSize((pairCount + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);
    Yuy2ToBgraKernel<<<gridSize, blockSize, 0, stream>>>(dst, dstPitchBytes,
                                                         src, srcPitchBytes,
                                                         width, height);
    CheckCuda("Yuy2ToBgraKernel launch failed");
}

void LaunchRotateQuarterLinear(uchar4* dst, size_t dstPitchBytes,
                               const uchar4* src, size_t srcPitchBytes,
                               int srcWidth, int srcHeight,
                               int quarterTurnsClockwise,
                               cudaStream_t stream) {
    if (srcWidth <= 0 || srcHeight <= 0 ||
        quarterTurnsClockwise < 1 || quarterTurnsClockwise > 3) {
        return;
    }
    const bool swapExtent = (quarterTurnsClockwise & 1) != 0;
    const int dstWidth = swapExtent ? srcHeight : srcWidth;
    const int dstHeight = swapExtent ? srcWidth : srcHeight;
    const dim3 blockSize(16, 16);
    const dim3 gridSize((dstWidth + blockSize.x - 1) / blockSize.x,
                        (dstHeight + blockSize.y - 1) / blockSize.y);
    RotateQuarterKernel<<<gridSize, blockSize, 0, stream>>>(dst, dstPitchBytes,
                                                            src, srcPitchBytes,
                                                            srcWidth, srcHeight,
                                                            dstWidth, dstHeight,
                                                            quarterTurnsClockwise);
    CheckCuda("RotateQuarterKernel launch failed");
}

void LaunchKeystoneWarp(uchar4* dst, size_t dstPitchBytes,
                        const uchar4* src, size_t srcPitchBytes,
                        int width, int height,
                        const float homography[9],
                        cudaStream_t stream) {
    if (width <= 0 || height <= 0 || homography == nullptr) {
        return;
    }
    KeystoneHomography h{};
    for (int i = 0; i < 9; ++i) {
        h.m[i] = homography[i];
    }
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);
    KeystoneWarpKernel<<<gridSize, blockSize, 0, stream>>>(dst, dstPitchBytes,
                                                           src, srcPitchBytes,
                                                           width, height,
                                                           h);
    CheckCuda("KeystoneWarpKernel launch failed");
}

void LaunchAutoContrastHistogram(unsigned int* histogram256,
                                 const uchar4* src, size_t srcPitchBytes,
                                 int width, int height,
                                 cudaStream_t stream) {
    if (width <= 0 || height <= 0) {
        return;
    }
    CheckCudaStatus(cudaMemsetAsync(histogram256, 0, 256 * sizeof(unsigned int), stream),
                    "cudaMemsetAsync histogram failed");
    // 16x16 = 256 threads per block: one shared bin per thread.
    const dim3 blockSize(16, 16);
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                        (height + blockSize.y - 1) / blockSize.y);
    AutoContrastHistogramKernel<<<gridSize, blockSize, 0, stream>>>(histogram256,
                                                                    src, srcPitchBytes,
                                                                    width, height);
    CheckCuda("AutoContrastHistogramKernel launch failed");
}

void LaunchAutoContrastAnalysis(const unsigned int* histogram256,
                                int pixelCount,
                                float2* levels,
                                bool levelsValid,
                                cudaStream_t stream) {
    if (pixelCount <= 0) {
        return;
    }
    AutoContrastAnalysisKernel<<<1, 1, 0, stream>>>(histogram256,
                                                    pixelCount,
                                                    levels,
                                                    levelsValid ? 1 : 0);
    CheckCuda("AutoContrastAnalysisKernel launch failed");
}

bool UploadGaussianKernel(int radius, float sigma, cudaStream_t stream) {
    if (radius <= 0 || sigma <= 0.0f) {
        return false;
    }

    const int clampedRadius = std::min(std::max(radius, 1), kMaxBlurRadius);
    const float sigmaClamped = std::max(sigma, 0.001f);
    const int kernelSize = clampedRadius * 2 + 1;

    std::vector<float> kernel(static_cast<size_t>(kernelSize));
    const float sigma2 = sigmaClamped * sigmaClamped;
    const float denom = 2.0f * sigma2;
    float weightSum = 0.0f;
    for (int i = -clampedRadius; i <= clampedRadius; ++i) {
        const float x = static_cast<float>(i);
        const float weight = std::exp(-(x * x) / denom);
        kernel[static_cast<size_t>(i + clampedRadius)] = weight;
        weightSum += weight;
    }
    if (weightSum <= 0.0f) {
        return false;
    }
    for (float& weight : kernel) {
        weight /= weightSum;
    }

    cudaError_t status = cudaMemcpyToSymbol(gGaussianKernel, kernel.data(), kernel.size() * sizeof(float),
                                            0, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return false;
    }
    status = cudaMemcpyToSymbol(gGaussianRadius, &clampedRadius, sizeof(int),
                                0, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return false;
    }
    status = cudaMemcpyToSymbol(gGaussianKernelSize, &kernelSize, sizeof(int),
                                0, cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        return false;
    }
    status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        return false;
    }
    return true;
}

} // namespace openzoom

#endif // _WIN32
