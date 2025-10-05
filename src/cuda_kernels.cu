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

} // namespace openzoom

#endif // _WIN32
