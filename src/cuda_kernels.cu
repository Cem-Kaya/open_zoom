#ifdef _WIN32

#include "openzoom/cuda_kernels.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
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

namespace openzoom {

namespace {

__device__ unsigned char FloatToByte(float value) {
    const float clamped = fminf(fmaxf(value, 0.0f), 1.0f);
    return static_cast<unsigned char>(clamped * 255.0f);
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

constexpr int kMaxBlurRadius = 15;
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
