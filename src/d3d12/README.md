# d3d12 module

This module owns the Direct3D 12 presentation path.

Current responsibilities:
- select a hardware adapter and create the D3D12 device
- keep the swap chain equal to the render HWND's native client size and manage
  its frame-latency waitable object, per-frame resources, and upload buffers
- present CPU-generated BGRA frames
- draw persistent CPU/GPU scene textures through the canonical aspect-safe
  Fill/Fit transform using a full-screen triangle and bilinear sampler
- read back GPU textures for recording and capture, returning stable request
  ids so processed frames can be paired with their original camera frames
