# Generated results

Smoke tests write timestamped JSON reports under `results/smoke/`. Performance benchmarks
will write JSON and CSV reports under `results/benchmarks/`.

Each smoke report contains the fixed input path and one result per selected model,
including status, runtime versions, load time, inference time, peak CUDA memory, a short
prediction summary, and generated-token count for VLMs. Failed results include a stable
error type and message.

Generated reports are ignored by Git. This file documents and retains the directory
layout without committing device-specific benchmark output.
