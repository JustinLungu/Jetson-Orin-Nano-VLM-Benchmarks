# Generated results

Limited benchmarks write reports under `results/benchmarks/`, while complete-dataset runs
use `results/benchmarks/full_run/`.

Each benchmark report contains its model, dataset, precision, software environment,
per-image inference status and latency, final latency/throughput
summary, and compact Jetson RAM, CUDA-memory, swap, power, and temperature measurements.
It is atomically replaced after every image. `run_status` identifies running, completed,
interrupted, and failed reports.

Generated reports are ignored by Git. This file documents and retains the directory
layout without committing device-specific benchmark output.
