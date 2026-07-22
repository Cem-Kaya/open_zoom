# common module

This module contains shared processing and output helpers that are reused across the app.

Current contents:
- asynchronous assistive analysis in `assistive_runtime.cpp`, including shared
  response instructions for Codex and OpenAI-compatible requests
- the permission-aware native `codex app-server` JSON-RPC transport in
  `codex_app_server_client.cpp`, with model/reasoning selection and guarded
  user-configured response preferences
- CPU frame conversion and image effects in `image_processing.cpp`
- stage orchestration and debug compositing in `frame_pipeline.cpp`
- Media Foundation sink-writer recording in `media_writer.cpp`

If additional cross-cutting helpers land later, keep them small and explicitly reusable rather than turning this directory into a misc bucket.
