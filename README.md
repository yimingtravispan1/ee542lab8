# EE542 Lab 8

CNN inference on network packets using TensorRT, DPDK, and NVIDIA DOCA GPUNetIO. Includes image packet generation and reassembly, GPU classification, and GPU/CPU RDMA benchmarks on CloudLab.

See [report.md](report.md) for the setup, results, and limitations. Parts 1–6 used a local RTX 5070 under WSL2/Docker Desktop; Part 7 used two CloudLab A30/ConnectX-6 hosts.

| Path | Contents |
| --- | --- |
| [src/](src/) | Model export, inference, packet generation, and DPDK receiver |
| [logs/](logs/) | Experiment output; Part 7 logs in `logs/part7/` |
| [results/](results/) | Benchmark and verification data; Part 7 CSVs in `results/part7/` |
| [figures/part7/](figures/part7/) | RDMA latency and bandwidth plots |
| [CloudLab profile](https://github.com/lanluolan/lab8-cloudlab-profile) | Two-node CloudLab experiment profile |
| [requirements.txt](requirements.txt) | Recorded Python dependencies |
| [meson.part6.build](meson.part6.build) | DOCA sample build configuration for the RTX 5070 |
