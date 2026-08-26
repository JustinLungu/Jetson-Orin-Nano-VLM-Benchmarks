# Generated results

Smoke tests write timestamped JSON reports under `results/smoke/`. Limited performance
benchmarks write reports under `results/benchmarks/`, while complete-dataset runs use
`results/benchmarks/full_run/`.

Each smoke report contains the fixed input path and one result per selected model,
including status, runtime versions, load time, inference time, peak CUDA memory, a short
prediction summary, and generated-token count for VLMs. Failed results include a stable
error type and message.

Each benchmark report contains its model, dataset, precision, checkpoint identity,
software environment, per-image inference status and latency, final latency/throughput
summary, and compact Jetson RAM, CUDA-memory, swap, power, and temperature measurements.
It is atomically replaced after every image. `run_status` identifies running, completed,
interrupted, and failed reports.

Generated reports are ignored by Git. This file documents and retains the directory
layout without committing device-specific benchmark output.
