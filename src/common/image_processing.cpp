#ifdef _WIN32

#include "openzoom/common/image_processing.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace openzoom {
namespace processing {

namespace {

inline int ClampToByte(int value)
{
    if (value < 0) {
        return 0;
    }
    if (value > 255) {
        return 255;
    }
    return value;
}

} // namespace

void CopyArgbToBgra(const uint8_t* src,
                    UINT srcStride,
                    UINT width,
                    UINT height,
                    std::vector<uint8_t>& dst)
{
    const UINT dstStride = width * 4;
    dst.resize(static_cast<size_t>(dstStride) * height);
    for (UINT y = 0; y < height; ++y) {
        const uint8_t* srcRow = src + static_cast<size_t>(y) * srcStride;
        uint8_t* dstRow = dst.data() + static_cast<size_t>(y) * dstStride;
        std::memcpy(dstRow, srcRow, dstStride);
    }
}

void CopyRgbxToBgra(const uint8_t* src,
                    UINT srcStride,
                    UINT width,
                    UINT height,
                    std::vector<uint8_t>& dst)
{
    const UINT dstStride = width * 4;
    dst.resize(static_cast<size_t>(dstStride) * height);
    for (UINT y = 0; y < height; ++y) {
        const uint8_t* srcRow = src + static_cast<size_t>(y) * srcStride;
        uint8_t* dstRow = dst.data() + static_cast<size_t>(y) * dstStride;
        for (UINT x = 0; x < width; ++x) {
            dstRow[x * 4 + 0] = srcRow[x * 4 + 0];
            dstRow[x * 4 + 1] = srcRow[x * 4 + 1];
            dstRow[x * 4 + 2] = srcRow[x * 4 + 2];
            dstRow[x * 4 + 3] = 255;
        }
    }
}

bool ConvertNv12ToBgra(const uint8_t* src,
                       size_t srcSize,
                       UINT strideY,
                       UINT width,
                       UINT height,
                       std::vector<uint8_t>& dst)
{
    if (!src || width == 0 || height == 0) {
        return false;
    }

    if (strideY == 0) {
        strideY = width;
    }

    const size_t yPlaneSize = static_cast<size_t>(strideY) * height;
    const size_t uvStride = strideY;
    const size_t uvPlaneSize = uvStride * ((height + 1) / 2);

    if (srcSize < yPlaneSize + uvPlaneSize) {
        return false;
    }

    dst.resize(static_cast<size_t>(width) * height * 4);

    const uint8_t* yPlane = src;
    const uint8_t* uvPlane = src + yPlaneSize;

    for (UINT y = 0; y < height; ++y) {
        const uint8_t* yRow = yPlane + static_cast<size_t>(y) * strideY;
        const uint8_t* uvRow = uvPlane + static_cast<size_t>(y / 2) * uvStride;
        uint8_t* dstRow = dst.data() + static_cast<size_t>(y) * width * 4;

        for (UINT x = 0; x < width; ++x) {
            const int yValue = yRow[x];
            const int uvIndex = (x / 2) * 2;
            const int uValue = uvRow[uvIndex] - 128;
            const int vValue = uvRow[uvIndex + 1] - 128;

            int c = yValue - 16;
            if (c < 0) {
                c = 0;
            }

            const int d = uValue;
            const int e = vValue;

            const int r = (298 * c + 409 * e + 128) >> 8;
            const int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
            const int b = (298 * c + 516 * d + 128) >> 8;

            const UINT dstIndex = x * 4;
            dstRow[dstIndex + 0] = static_cast<uint8_t>(ClampToByte(b));
            dstRow[dstIndex + 1] = static_cast<uint8_t>(ClampToByte(g));
            dstRow[dstIndex + 2] = static_cast<uint8_t>(ClampToByte(r));
            dstRow[dstIndex + 3] = 255;
        }
    }

    return true;
}

bool ConvertYuy2ToBgra(const uint8_t* src,
                       size_t srcSize,
                       UINT stride,
                       UINT width,
                       UINT height,
                       std::vector<uint8_t>& dst)
{
    if (!src || width == 0 || height == 0) {
        return false;
    }

    if (stride == 0) {
        stride = width * 2;
    }

    const size_t required = static_cast<size_t>(stride) * height;
    if (srcSize < required) {
        return false;
    }

    dst.resize(static_cast<size_t>(width) * height * 4);

    for (UINT y = 0; y < height; ++y) {
        const uint8_t* srcRow = src + static_cast<size_t>(y) * stride;
        uint8_t* dstRow = dst.data() + static_cast<size_t>(y) * width * 4;

        for (UINT x = 0; x < width; x += 2) {
            const UINT srcIndex = x * 2;
            if (srcIndex + 3 >= stride) {
                break;
            }

            const int y0 = srcRow[srcIndex + 0];
            const int u = srcRow[srcIndex + 1] - 128;
            const int y1 = srcRow[srcIndex + 2];
            const int v = srcRow[srcIndex + 3] - 128;

            const int c0 = y0 - 16;
            const int c1 = y1 - 16;

            const int r0 = (298 * c0 + 409 * v + 128) >> 8;
            const int g0 = (298 * c0 - 100 * u - 208 * v + 128) >> 8;
            const int b0 = (298 * c0 + 516 * u + 128) >> 8;

            const int r1 = (298 * c1 + 409 * v + 128) >> 8;
            const int g1 = (298 * c1 - 100 * u - 208 * v + 128) >> 8;
            const int b1 = (298 * c1 + 516 * u + 128) >> 8;

            const UINT dstIndex0 = x * 4;
            dstRow[dstIndex0 + 0] = static_cast<uint8_t>(ClampToByte(b0));
            dstRow[dstIndex0 + 1] = static_cast<uint8_t>(ClampToByte(g0));
            dstRow[dstIndex0 + 2] = static_cast<uint8_t>(ClampToByte(r0));
            dstRow[dstIndex0 + 3] = 255;

            dstRow[dstIndex0 + 4] = static_cast<uint8_t>(ClampToByte(b1));
            dstRow[dstIndex0 + 5] = static_cast<uint8_t>(ClampToByte(g1));
            dstRow[dstIndex0 + 6] = static_cast<uint8_t>(ClampToByte(r1));
            dstRow[dstIndex0 + 7] = 255;
        }
    }

    return true;
}

void ApplyBlackWhite(const std::vector<uint8_t>& src,
                     std::vector<uint8_t>& dst,
                     float threshold)
{
    if (dst.size() != src.size()) {
        dst.resize(src.size());
    }
    const float thresh = std::clamp(threshold, 0.0f, 1.0f);
    for (size_t i = 0; i < src.size(); i += 4) {
        const float r = src[i + 2] / 255.0f;
        const float g = src[i + 1] / 255.0f;
        const float b = src[i + 0] / 255.0f;
        const float luminance = 0.299f * r + 0.587f * g + 0.114f * b;
        const uint8_t value = luminance >= thresh ? 255 : 0;
        dst[i + 0] = value;
        dst[i + 1] = value;
        dst[i + 2] = value;
        dst[i + 3] = src[i + 3];
    }
}

void ApplyZoom(const std::vector<uint8_t>& src,
               std::vector<uint8_t>& dst,
               UINT width,
               UINT height,
               float zoomAmount,
               float centerXNormalized,
               float centerYNormalized)
{
    const float zoom = std::max(1.0f, zoomAmount);
    const UINT stride = width * 4;
    if (dst.size() != src.size()) {
        dst.resize(src.size());
    }

    const float maxIndexX = static_cast<float>(width > 0 ? width - 1 : 0);
    const float maxIndexY = static_cast<float>(height > 0 ? height - 1 : 0);
    const float outputCenterX = maxIndexX * 0.5f;
    const float outputCenterY = maxIndexY * 0.5f;

    float centerX = std::clamp(centerXNormalized, 0.0f, 1.0f) * maxIndexX;
    float centerY = std::clamp(centerYNormalized, 0.0f, 1.0f) * maxIndexY;

    const float halfVisibleWidth = (static_cast<float>(width)) / (zoom * 2.0f);
    const float halfVisibleHeight = (static_cast<float>(height)) / (zoom * 2.0f);

    if (width > 1) {
        const float minCenterX = std::max(0.0f, halfVisibleWidth - 0.5f);
        const float maxCenterX = std::min(maxIndexX, static_cast<float>(width) - 1.0f - (halfVisibleWidth - 0.5f));
        if (minCenterX <= maxCenterX) {
            centerX = std::clamp(centerX, minCenterX, maxCenterX);
        }
    }

    if (height > 1) {
        const float minCenterY = std::max(0.0f, halfVisibleHeight - 0.5f);
        const float maxCenterY = std::min(maxIndexY, static_cast<float>(height) - 1.0f - (halfVisibleHeight - 0.5f));
        if (minCenterY <= maxCenterY) {
            centerY = std::clamp(centerY, minCenterY, maxCenterY);
        }
    }

    for (UINT y = 0; y < height; ++y) {
        uint8_t* dstRow = dst.data() + static_cast<size_t>(y) * stride;
        for (UINT x = 0; x < width; ++x) {
            const float sx = (static_cast<float>(x) - outputCenterX) / zoom + centerX;
            const float sy = (static_cast<float>(y) - outputCenterY) / zoom + centerY;

            int sampleX = static_cast<int>(std::lroundf(sx));
            int sampleY = static_cast<int>(std::lroundf(sy));
            sampleX = std::clamp(sampleX, 0, static_cast<int>(width) - 1);
            sampleY = std::clamp(sampleY, 0, static_cast<int>(height) - 1);

            const uint8_t* srcPixel = src.data() + static_cast<size_t>(sampleY) * stride + sampleX * 4;
            uint8_t* dstPixel = dstRow + x * 4;
            dstPixel[0] = srcPixel[0];
            dstPixel[1] = srcPixel[1];
            dstPixel[2] = srcPixel[2];
            dstPixel[3] = srcPixel[3];
        }
    }
}

void ApplyGaussianBlur(const std::vector<uint8_t>& src,
                       std::vector<uint8_t>& scratch,
                       std::vector<uint8_t>& dst,
                       UINT width,
                       UINT height,
                       int radius,
                       float sigma)
{
    if (radius <= 0 || sigma <= 0.0f || src.empty() || width == 0 || height == 0) {
        dst = src;
        return;
    }

    const size_t pixelCount = static_cast<size_t>(width) * height;
    scratch.resize(pixelCount * 4);
    dst.resize(pixelCount * 4);

    const int kernelSize = radius * 2 + 1;
    std::vector<float> kernel(static_cast<size_t>(kernelSize));
    const float sigma2 = sigma * sigma;
    const float denom = 2.0f * sigma2;
    float weightSum = 0.0f;
    for (int i = -radius; i <= radius; ++i) {
        const float x = static_cast<float>(i);
        const float weight = std::exp(-(x * x) / denom);
        kernel[static_cast<size_t>(i + radius)] = weight;
        weightSum += weight;
    }
    if (weightSum <= 0.0f) {
        dst = src;
        return;
    }
    for (float& weight : kernel) {
        weight /= weightSum;
    }

    auto samplePixel = [&](const std::vector<uint8_t>& buffer, UINT x, UINT y) -> const uint8_t* {
        return buffer.data() + (static_cast<size_t>(y) * width + x) * 4;
    };

    auto writePixel = [](std::vector<uint8_t>& buffer, UINT width, UINT x, UINT y,
                         float b, float g, float r, float a) {
        uint8_t* dstPixel = buffer.data() + (static_cast<size_t>(y) * width + x) * 4;
        dstPixel[0] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(std::lround(b)), 0, 255));
        dstPixel[1] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(std::lround(g)), 0, 255));
        dstPixel[2] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(std::lround(r)), 0, 255));
        dstPixel[3] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(std::lround(a)), 0, 255));
    };

    // Horizontal pass
    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            float accumB = 0.0f;
            float accumG = 0.0f;
            float accumR = 0.0f;
            float accumA = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int sampleX = static_cast<int>(x) + k;
                sampleX = std::clamp(sampleX, 0, static_cast<int>(width) - 1);
                const uint8_t* srcPixel = samplePixel(src, static_cast<UINT>(sampleX), y);
                const float weight = kernel[static_cast<size_t>(k + radius)];
                accumB += weight * srcPixel[0];
                accumG += weight * srcPixel[1];
                accumR += weight * srcPixel[2];
                accumA += weight * srcPixel[3];
            }
            writePixel(scratch, width, x, y, accumB, accumG, accumR, accumA);
        }
    }

    // Vertical pass
    for (UINT y = 0; y < height; ++y) {
        for (UINT x = 0; x < width; ++x) {
            float accumB = 0.0f;
            float accumG = 0.0f;
            float accumR = 0.0f;
            float accumA = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int sampleY = static_cast<int>(y) + k;
                sampleY = std::clamp(sampleY, 0, static_cast<int>(height) - 1);
                const uint8_t* srcPixel = samplePixel(scratch, x, static_cast<UINT>(sampleY));
                const float weight = kernel[static_cast<size_t>(k + radius)];
                accumB += weight * srcPixel[0];
                accumG += weight * srcPixel[1];
                accumR += weight * srcPixel[2];
                accumA += weight * srcPixel[3];
            }
            writePixel(dst, width, x, y, accumB, accumG, accumR, accumA);
        }
    }
}

void ApplyTemporalSmoothCpu(std::vector<uint8_t>& frame,
                            std::vector<float>& temporalHistory,
                            bool& historyValid,
                            UINT width,
                            UINT height,
                            float temporalSmoothAlpha,
                            bool temporalSmoothEnabled)
{
    if (!temporalSmoothEnabled) {
        historyValid = false;
        temporalHistory.clear();
        return;
    }

    if (frame.empty() || width == 0 || height == 0) {
        return;
    }

    const size_t channelCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
    if (frame.size() < channelCount) {
        return;
    }

    if (temporalHistory.size() != channelCount) {
        temporalHistory.assign(channelCount, 0.0f);
        historyValid = false;
    }

    const float alphaClamped = std::clamp(temporalSmoothAlpha, 0.0f, 1.0f);
    const float oneMinusAlpha = 1.0f - alphaClamped;

    if (!historyValid) {
        for (size_t i = 0; i < channelCount; ++i) {
            temporalHistory[i] = static_cast<float>(frame[i]);
        }
        historyValid = true;
        return;
    }

    for (size_t i = 0; i < channelCount; ++i) {
        if ((i & 3u) == 3u) {
            temporalHistory[i] = 255.0f;
            frame[i] = 255u;
            continue;
        }

        const float curr = static_cast<float>(frame[i]);
        const float prev = temporalHistory[i];
        const float blended = alphaClamped * curr + oneMinusAlpha * prev;
        temporalHistory[i] = blended;
        frame[i] = static_cast<uint8_t>(std::clamp<int>(static_cast<int>(std::lround(blended)), 0, 255));
    }
}

} // namespace processing
} // namespace openzoom

#endif // _WIN32
