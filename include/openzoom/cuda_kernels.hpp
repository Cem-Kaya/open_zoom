#pragma once

#ifdef _WIN32

#include <cuda_runtime_api.h>

namespace openzoom {

void LaunchGradientKernel(cudaSurfaceObject_t surface, int width, int height, float timeSeconds);
void LaunchBlackWhiteKernel(cudaSurfaceObject_t surface, int width, int height, float threshold);
void LaunchZoomKernel(cudaSurfaceObject_t surface, int width, int height, float zoomAmount);

} // namespace openzoom

#endif // _WIN32
