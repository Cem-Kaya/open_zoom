# common module

Shared utilities (logging, math helpers, feature flags) and the CPU frame
pipeline live here so other modules stay lean. The `CpuFramePipeline` class
now owns conversion, rotation, temporal history, and debug compositing for the
CPU fallback path.
