# capture module

This module owns Media Foundation camera discovery and frame acquisition.

Current responsibilities:
- enumerate available video devices
- list native capture modes for the selected device
- stream frames on a dedicated capture thread via `IMFSourceReader`
- own each `IMFActivate` session through `ShutdownObject()` so mode probes and capture restarts do not reuse a shut-down media source
- report capture-thread failures and adapt frame metadata when a device changes its current media type

Frame conversion into BGRA and downstream effects happen in `src/common/`, not here.
