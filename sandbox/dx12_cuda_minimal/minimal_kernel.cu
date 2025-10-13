#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <math.h>

namespace {

__global__ void FillBufferKernel(uchar4* destination,
                                 size_t pitchInBytes,
                                 unsigned int width,
                                 unsigned int height,
                                 float t)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const float fx = static_cast<float>(x) / static_cast<float>(width);
    const float fy = static_cast<float>(y) / static_cast<float>(height);

    const unsigned char r = static_cast<unsigned char>(255.0f * fx);
    const unsigned char g = static_cast<unsigned char>(255.0f * fy);
    const unsigned char b = static_cast<unsigned char>(255.0f * fmodf(t, 1.0f));

    const uchar4 pixel = make_uchar4(r, g, b, 255);
    auto rowPtr = reinterpret_cast<uchar4*>(reinterpret_cast<unsigned char*>(destination) + y * pitchInBytes);
    rowPtr[x] = pixel;
}

} // namespace

extern "C" void LaunchFillBuffer(uchar4* destination,
                                 size_t pitch,
                                 unsigned int width,
                                 unsigned int height,
                                 cudaStream_t stream)
{
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y,
                 1);

    const float timeSeconds = 0.5f;
    FillBufferKernel<<<gridDim, blockDim, 0, stream>>>(destination, pitch, width, height, timeSeconds);
}
