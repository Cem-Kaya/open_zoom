# OpenZoom Tests

The hardware-independent targets cover settings persistence, canonical
Fill/Fit viewport geometry, cached SuperRes ROI registration, and monotonic
CUDA/D3D12 fence sequencing. Run them with the CPU-only Windows preset:

```bat
cmake --preset msvc-cpu
cmake --build --preset msvc-cpu-build
ctest --preset msvc-cpu-tests --no-tests=error
```

Camera capture, D3D12 presentation, CUDA processing, and Media Foundation
recording still require manual testing on Windows hardware.
