# Third-Party Licenses and Attributions

OpenZoom relies on several third-party SDKs and code drops. Keep this summary with any redistributed build and update it whenever dependencies or bundled notices change.

## Qt 6
- Upstream: <https://www.qt.io/>
- License family: LGPL-3.0 / GPL-3.0 / commercial, depending on how Qt is obtained
- OpenZoom usage: dynamically linked Qt Widgets runtime deployed via `windeployqt`
- Notes:
  - If you redistribute Qt DLLs, include the required Qt notices and license text that apply to your distribution.
  - If you modify Qt itself, those changes must be handled under Qt's licensing terms.
  - Development builds expect the user to provide a local Qt installation.

## NVIDIA Image Scaling
- Upstream: <https://github.com/NVIDIAGameWorks/NVIDIAImageScaling>
- Local notice file: [`third_party/nvidia_nis/LICENSE.txt`](../third_party/nvidia_nis/LICENSE.txt)
- Usage: optional CUDA spatial sharpening path
- Attribution: retain NVIDIA's notice when redistributing modified or bundled source material

## AMD FidelityFX Super Resolution 1.0
- Upstream: <https://github.com/GPUOpen-Effects/FidelityFX-SuperResolution>
- Local notice file: [`third_party/amd_fsr1/LICENSE.txt`](../third_party/amd_fsr1/LICENSE.txt)
- License: MIT
- Usage: optional FSR-style CUDA spatial sharpening path
- Attribution: preserve the MIT license text and AMD attribution in redistributed source packages

## NVIDIA CUDA Toolkit
- Upstream: <https://developer.nvidia.com/cuda-toolkit>
- License: NVIDIA CUDA Toolkit EULA
- OpenZoom redistribution scope:
  - runtime DLLs only
  - no CUDA headers, compilers, or developer tools are bundled
- Review NVIDIA's current redistribution terms before shipping binaries that include CUDA runtime files

## Microsoft Platform Components
- APIs used: Media Foundation, Direct3D 12, DXGI, Windows Imaging Component, and other Windows SDK libraries
- License source: Windows SDK / Visual Studio / OS redistribution terms
- Notes: these platform APIs are not copied into the repository as standalone third-party source code

## Not Currently Bundled
- OpenCV DNN
- TensorRT
- external AI model weights

Those remain roadmap items. If they are added later, this file must be expanded in the same change.

## Optional External Tools And Services
- Tesseract OCR may be used at runtime if installed locally, but it is not bundled by this repository or the current Windows packaging scripts.
- OpenAI-compatible VLM services may be used at runtime through user-supplied endpoint credentials, but no hosted model or service SDK is bundled here.
