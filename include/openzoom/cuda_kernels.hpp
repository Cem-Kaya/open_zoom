#pragma once

#ifdef _WIN32

#include <cuda_runtime_api.h>

namespace openzoom {

void LaunchGradientKernel(cudaSurfaceObject_t surface, int width, int height, float timeSeconds);
void LaunchBlackWhiteKernel(cudaSurfaceObject_t surface, int width, int height, float threshold);
void LaunchZoomKernel(cudaSurfaceObject_t surface, int width, int height, float zoomAmount);

void LaunchBlackWhiteLinear(uchar4* dst, size_t dstPitchBytes,
                            const uchar4* src, size_t srcPitchBytes,
                            int width, int height, float threshold, cudaStream_t stream);

void LaunchZoomLinear(uchar4* dst, size_t dstPitchBytes,
                      const uchar4* src, size_t srcPitchBytes,
                      int width, int height,
                      float zoomAmount, float centerXNorm, float centerYNorm,
                      cudaStream_t stream);

void LaunchGaussianBlurLinear(uchar4* dst, size_t dstPitchBytes,
                              uchar4* scratch, size_t scratchPitchBytes,
                              const uchar4* src, size_t srcPitchBytes,
                              int width, int height,
                              cudaStream_t stream);

void LaunchFocusMarkerLinear(uchar4* buffer, size_t pitchBytes,
                             int width, int height,
                             float centerXNorm, float centerYNorm,
                             cudaStream_t stream);

bool UploadGaussianKernel(int radius, float sigma, cudaStream_t stream);

} // namespace openzoom

#endif // _WIN32
