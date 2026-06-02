# capture module

This module owns Media Foundation camera discovery and frame acquisition.

Current responsibilities:
- enumerate available video devices
- list native capture modes for the selected device
- stream frames on a dedicated capture thread via `IMFSourceReader`

Frame conversion into BGRA and downstream effects happen in `src/common/`, not here.
