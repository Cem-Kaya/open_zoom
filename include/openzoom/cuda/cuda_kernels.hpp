#pragma once

#ifdef _WIN32

#include <cuda_runtime_api.h>

#include <cstdint>

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

void LaunchBlendLinear(uchar4* dst, size_t dstPitchBytes,
                       const uchar4* base, size_t basePitchBytes,
                       const uchar4* enhanced, size_t enhancedPitchBytes,
                       int width, int height, float strength,
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

void LaunchFsrEasuRcasLinear(uchar4* dst, size_t dstPitchBytes,
                             const uchar4* src, size_t srcPitchBytes,
                             int srcWidth, int srcHeight,
                             int dstWidth, int dstHeight,
                             float sharpness,
                             cudaStream_t stream);

void LaunchNisLinear(uchar4* dst, size_t dstPitchBytes,
                     const uchar4* src, size_t srcPitchBytes,
                     int srcWidth, int srcHeight,
                     int dstWidth, int dstHeight,
                     float sharpness,
                     cudaStream_t stream);

void LaunchTemporalSmoothLinear(uchar4* dst, size_t dstPitchBytes,
                                const uchar4* src, size_t srcPitchBytes,
                                float4* history, size_t historyPitchBytes,
                                int width, int height,
                                float alpha,
                                bool historyValid,
                                cudaStream_t stream);

// Device-resident motion state for video stabilization. It is never read back
// to the host: the estimate kernel updates it and the warp kernel consumes it.
struct StabilizationState {
    float2 actualPath;   // accumulated camera translation, full-res pixels
    float2 smoothPath;   // low-pass filtered path, full-res pixels
    float2 correction;   // smoothPath - actualPath, clamped to the warp margin
};

void LaunchStabilizationLumaDownsample(float* dstLuma,
                                       int smallWidth, int smallHeight,
                                       const uchar4* src, size_t srcPitchBytes,
                                       int width, int height,
                                       int factorX, int factorY,
                                       cudaStream_t stream);

void LaunchStabilizationProjections(const float* luma,
                                    int smallWidth, int smallHeight,
                                    float* colProj, float* rowProj,
                                    cudaStream_t stream);

void LaunchStabilizationEstimate(const float* currColProj, const float* currRowProj,
                                 float* prevColProj, float* prevRowProj,
                                 int smallWidth, int smallHeight,
                                 float factorX, float factorY,
                                 int fullWidth, int fullHeight,
                                 float strength,
                                 bool previousValid,
                                 StabilizationState* state,
                                 cudaStream_t stream);

void LaunchStabilizationWarp(uchar4* dst, size_t dstPitchBytes,
                             const uchar4* src, size_t srcPitchBytes,
                             int width, int height,
                             const StabilizationState* state,
                             cudaStream_t stream);

// autoContrastLevels: device float2 (x=lo, y=hi, normalized 0..1) written by
// LaunchAutoContrastAnalysis; pass nullptr to disable the auto-contrast remap.
void LaunchDisplayColorGradeLinear(uchar4* buffer, size_t pitchBytes,
                                   int width, int height,
                                   int colorTransform, float contrast, float brightness,
                                   const float2* autoContrastLevels,
                                   float autoContrastStrength,
                                   cudaStream_t stream);

// BT.601 limited-range YUV -> BGRA, integer math identical to the CPU
// converters in src/common/image_processing.cpp.
void LaunchNv12ToBgraLinear(uchar4* dst, size_t dstPitchBytes,
                            const unsigned char* yPlane, size_t yPitchBytes,
                            const unsigned char* uvPlane, size_t uvPitchBytes,
                            int width, int height,
                            cudaStream_t stream);

void LaunchYuy2ToBgraLinear(uchar4* dst, size_t dstPitchBytes,
                            const unsigned char* src, size_t srcPitchBytes,
                            int width, int height,
                            cudaStream_t stream);

// Rotate by quarterTurnsClockwise in {1,2,3}. For 1 and 3 the destination is
// srcHeight x srcWidth; for 2 it matches the source extent.
void LaunchRotateQuarterLinear(uchar4* dst, size_t dstPitchBytes,
                               const uchar4* src, size_t srcPitchBytes,
                               int srcWidth, int srcHeight,
                               int quarterTurnsClockwise,
                               cudaStream_t stream);

// homography maps OUTPUT pixel coordinates to SOURCE pixel coordinates
// (row-major 3x3, uploaded per launch as kernel arguments). Samples bilinearly,
// black outside the source rect.
void LaunchKeystoneWarp(uchar4* dst, size_t dstPitchBytes,
                        const uchar4* src, size_t srcPitchBytes,
                        int width, int height,
                        const float homography[9],
                        cudaStream_t stream);

// Zeroes histogram256 (async) then accumulates a 256-bin luma histogram using
// shared-memory per-block histograms merged with global atomics.
void LaunchAutoContrastHistogram(unsigned int* histogram256,
                                 const uchar4* src, size_t srcPitchBytes,
                                 int width, int height,
                                 cudaStream_t stream);

// Single tiny block: derives the 2nd/98th percentile levels from the histogram
// and low-passes them into `levels` (device float2, normalized 0..1). With
// levelsValid false the new measurement seeds the state directly.
void LaunchAutoContrastAnalysis(const unsigned int* histogram256,
                                int pixelCount,
                                float2* levels,
                                bool levelsValid,
                                cudaStream_t stream);

// Text-clarity kernels share a full-resolution float workspace and byte masks.
// `analysis` is device-resident: x=auto light-text flag, y=scene class
// (0=mixed, 1=paper, 2=board), z=mean luma x10000, w=contrast x10000.
void LaunchTextLocalStatistics(float* luma, size_t floatPitchBytes,
                               float* horizontal, float* mean,
                               float* sqHorizontal, float* sqMean,
                               const uchar4* src, size_t srcPitchBytes,
                               int width, int height, int radius,
                               cudaStream_t stream);
void LaunchTextSceneAnalysis(const unsigned int* histogram256, int pixelCount,
                             int4* analysis, cudaStream_t stream);
void LaunchBackgroundFlattenLinear(uchar4* dst, size_t dstPitchBytes,
                                   const uchar4* src, size_t srcPitchBytes,
                                   const float* luma, const float* mean,
                                   size_t floatPitchBytes,
                                   int width, int height, float strength,
                                   bool suppressGlare, float glareStrength,
                                   const int4* analysis,
                                   cudaStream_t stream);
void LaunchClaheLinear(uchar4* dst, size_t dstPitchBytes,
                       const uchar4* src, size_t srcPitchBytes,
                       unsigned int* tileHistograms, float* tileMaps,
                       int width, int height, float clipLimit,
                       cudaStream_t stream);
void LaunchSauvolaMask(unsigned char* mask, size_t maskPitchBytes,
                       const float* luma, const float* mean, const float* sqMean,
                       size_t floatPitchBytes, int width, int height,
                       float strength, float softness, int polarityMode,
                       const int4* analysis, cudaStream_t stream);
void LaunchStrokeWeight(unsigned char* dst, size_t dstPitchBytes,
                        const unsigned char* src, size_t srcPitchBytes,
                        int width, int height, int strokeWeight,
                        cudaStream_t stream);
void LaunchTextMaskHysteresis(unsigned char* mask, size_t maskPitchBytes,
                              unsigned char* history, int width, int height,
                              float strength, bool historyValid,
                              cudaStream_t stream);
void LaunchTextMaskComposite(uchar4* dst, size_t dstPitchBytes,
                             const uchar4* src, size_t srcPitchBytes,
                             const unsigned char* mask, size_t maskPitchBytes,
                             int width, int height,
                             std::uint32_t foregroundBgra,
                             std::uint32_t backgroundBgra,
                             int compositeMode, const int4* analysis,
                             cudaStream_t stream);
void LaunchSmartSharpenLinear(uchar4* dst, size_t dstPitchBytes,
                              uchar4* scratch, size_t scratchPitchBytes,
                              const uchar4* src, size_t srcPitchBytes,
                              const unsigned char* mask, size_t maskPitchBytes,
                              int width, int height, float strength,
                              bool selective, cudaStream_t stream);
void LaunchFocusMetric(const float* luma, size_t floatPitchBytes,
                       int width, int height, float2* stats,
                       cudaStream_t stream);

bool UploadGaussianKernel(int radius, float sigma, cudaStream_t stream);
bool UploadDisplayColorLut(const std::uint32_t* lut256, cudaStream_t stream);

} // namespace openzoom

#endif // _WIN32
