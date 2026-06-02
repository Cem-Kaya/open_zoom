# d3d12 module

This module owns the Direct3D 12 presentation path.

Current responsibilities:
- select a hardware adapter and create the D3D12 device
- manage the swap chain and upload buffer
- present CPU-generated BGRA frames
- present GPU textures produced by CUDA
- read back GPU textures for recording and capture
