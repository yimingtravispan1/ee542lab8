# Laboratory #8 — Part 7 Report

**GPU-Memory RDMA with DOCA GPUNetIO on CloudLab**  
**Group / members:** [ADD BEFORE SUBMISSION]  
**Project / cluster:** `DOCA-GPU` / Wisconsin  

We tested GPU-memory RDMA between two CloudLab hosts, compared it with CPU-memory RDMA, and investigated the UDP receive sample and direct GPU doorbells. All successful GPUNetIO benchmarks used **`-p 1` CPU proxy**: CUDA builds requests, the CPU rings the NIC doorbell, and the NIC transfers data directly between GPU buffers.

## 1. Nodes and preparation

| Setting | node0 (server) | node1 (client) |
| --- | --- | --- |
| Experiment LAN | `10.10.1.1/24` | `10.10.1.2/24` |
| NIC / PCI address | `enp65s0np0` / `41:00.0` | Same |
| GPU / PCI address | NVIDIA A30, 24 GiB / `25:00.0` | Same |
| RDMA device | `rocep65s0` | Same |

Both hosts are identified as `d7525` in the experiment context; `profile.py` is not supplied. Both have the following configuration:

| Component | Observed configuration |
| --- | --- |
| OS / kernel | Ubuntu 24.04.5 LTS / `6.8.0-147-generic` |
| NVIDIA driver / module license | `580.196.07` / `Dual MIT/GPL` (open kernel module) |
| CUDA Toolkit | 13.0, `nvcc V13.0.88` |
| GDRCopy | `v2.5.2` at `/opt/mellanox/gdrcopy`; `gdrdrv` loaded |
| GPU persistence / hugepages | Enabled / 2048 × 2 MiB |
| NIC / driver / firmware | ConnectX-6 (`MT28908`) / `mlx5_core` / `20.38.1002 (DEL0000000010)` |
| Docker / storage | 29.1.3; data-root `/mydata/docker`; GPU access verified |
| GPU–NIC topology | `NODE`; GPU/NIC upstream bridges `20:03.1` / `40:01.1` |
| ACS / IOMMU after reboot | All shown bridge `ACSCtl` flags disabled; `/proc/cmdline` contains `amd_iommu=off iommu=off` |

`NODE` crosses PCIe host bridges within one NUMA node; the installation guide recommends `PIX`/`PXB` and avoiding `NODE`/`SYS`. The required comparison with the **actual Part 6 cloud instance remains pending**: its GPU, NIC/RDMA and topology records are not supplied.

After reboot, logs confirm hugepages, `gdrdrv`, persistence, ACS and the current RDMA device name. GPUNetIO was built for the A30:

```bash
make install install_examples PREFIX=$HOME/gpunetio/install CUDA_ARCH=80
export LD_LIBRARY_PATH="$HOME/gpunetio/install/lib:/opt/mellanox/gdrcopy/src:${LD_LIBRARY_PATH:-}"
export EX="$HOME/gpunetio/install/examples"
```

Both nodes have executable latency, put-bandwidth and write-bandwidth samples with no missing libraries in `ldd`; their successful runs support the build check.

## 2. DOCA UDP receive sample

On node0, the DOCA 3.5 development container built the sample for `sm_80` and ran:

```bash
/tmp/build/doca_gpunetio_simple_receive -n 41:00.0 -g 25:00.0 -e 0
```

DOCA `3.5.0098` initialized the GPU/NIC with per-thread reception (`-e 0`), then failed during DOCA Flow port creation:

```text
FW failed to execute general PRM command=0x100, status=BAD_PARAM_ERR (0x3)
Failed to query WQE based flow table capabilities.
Failed to get hws cap
Failed to start doca flow port with: Unknown error
```

The sample uses hardware steering to direct UDP packets into a GPU receive queue. The tested ConnectX-6/firmware rejected the required capability query. NVIDIA's [DOCA 3.5 GPUNetIO installation guide](https://networking-docs.nvidia.com/doca/archive/3-5-0/gpunetio-installation-and-setup) recommends **ConnectX-6 Dx or later adapters** with compatible firmware; our NIC is ConnectX-6, not Dx. This UDP steering failure does not prevent the RDMA Verbs tests below.

## 3. GPU-to-GPU latency (Step 7.8)

Start node0 first, then node1. `-l 3` selects the IPv4 RoCE v2 GID; `-p 1` enables CPU proxy.

```bash
# node0
DOCA_GPUNETIO_LOG=6 $EX/gpunetio_verbs_write_lat -g 25:00.0 -d rocep65s0 -l 3 -p 1
# node1
DOCA_GPUNETIO_LOG=6 $EX/gpunetio_verbs_write_lat -g 25:00.0 -d rocep65s0 -l 3 -p 1 -c 10.10.1.1
```

Both nodes completed successfully. Client results (2048 iterations per size) are:

| Message size | One-way latency (µs) |
| --- | ---: |
| 1 B | 8.39 |
| 64 B | 8.44 |
| 128 B | 8.52 |
| 256 B | 7.89 |
| 512 B | 7.48 |
| 1 KiB | 7.63 |
| 2 KiB | 7.86 |
| 4 KiB | 8.22 |
| 8 KiB | 9.11 |
| 16 KiB | 9.33 |

![GPUNetIO one-way latency versus message size](figures/gpunetio_latency.png)

The smallest payload (1 B) measured **8.39 µs** on node1 (8.38 µs on node0); the minimum latency was **7.48 µs at 512 B**. These are single-run measurements.

The **`-p 2` direct GPU-doorbell attempt** failed locally on both nodes:

```text
Function cudaHostRegister (err 800) failed
Can't export UAR to GPU.
```

Direct doorbells require the NIC's UAR/MMIO doorbell region to be mapped into GPU-accessible address space. This configuration could not export that mapping, so successful runs used CPU proxy. The node0 `-p 2` command also included `-c 10.10.1.1`; the logs demonstrate local setup failure, not a completed server/client benchmark.

## 4. GPU-to-GPU bandwidth

Each test used CPU proxy and 2048 iterations per size. `put_bw` uses GID 3; `write_bw` rejects `-l` and uses default GID 0 (link-local IPv6).

```bash
# put_bw: node0 server, then node1 client
DOCA_GPUNETIO_LOG=6 $EX/gpunetio_verbs_put_bw -g 25:00.0 -d rocep65s0 -l 3 -p 1
DOCA_GPUNETIO_LOG=6 $EX/gpunetio_verbs_put_bw -g 25:00.0 -d rocep65s0 -l 3 -p 1 -c 10.10.1.1
# write_bw: node0 server, then node1 client
DOCA_GPUNETIO_LOG=6 $EX/gpunetio_verbs_write_bw -g 25:00.0 -d rocep65s0 -p 1
DOCA_GPUNETIO_LOG=6 $EX/gpunetio_verbs_write_bw -g 25:00.0 -d rocep65s0 -p 1 -c 10.10.1.1
```

| Message size | put_bw, GID 3 (Gbps) | put_bw (Mpps) | write_bw, GID 0 (Gbps) | write_bw (Mpps) |
| --- | ---: | ---: | ---: | ---: |
| 1 B | 0.006499 | 0.812348 | 0.035955 | 4.494382 |
| 64 B | 0.416260 | 0.813008 | 2.301124 | 4.494382 |
| 128 B | 0.832182 | 0.812678 | 3.968993 | 3.875969 |
| 256 B | 1.665718 | 0.813339 | 9.002198 | 4.395605 |
| 512 B | 3.342309 | 0.815993 | 17.617203 | 4.301075 |
| 1 KiB | 6.681892 | 0.815661 | 33.368633 | 4.073319 |
| 2 KiB | 13.653333 | 0.833333 | 44.949246 | 2.743484 |
| 4 KiB | 29.114171 | 0.888494 | 51.200000 | 1.562500 |
| 8 KiB | 53.433344 | 0.815328 | 53.916910 | 0.822707 |
| 16 KiB | 56.557494 | 0.431499 | 54.050308 | 0.412371 |
| 32 KiB | 58.783273 | 0.224240 | 57.399603 | 0.218962 |
| 64 KiB | 46.755070 | 0.089178 | 45.544714 | 0.086870 |
| 128 KiB | 55.664284 | 0.053086 | 46.080111 | 0.043945 |
| 256 KiB | 55.904678 | 0.026657 | 55.926292 | 0.026668 |
| 512 KiB | 56.034255 | 0.013360 | 55.866327 | 0.013320 |
| 1 MiB | 55.975928 | 0.006673 | 55.948296 | 0.006670 |

![GPUNetIO bandwidth versus message size](figures/gpunetio_bandwidth.png)

At 64 KiB, `put_bw` reached **46.755070 Gbps / 0.089178 Mpps** and `write_bw` **45.544714 Gbps / 0.086870 Mpps**. Both peaked at 32 KiB: 58.783273 and 57.399603 Gbps, respectively. The cause of the dip at 64 KiB was not isolated.

**Data integrity remains unverified:** clients completed, but both node0 logs end after QP connection. Step 7.9 requires stopping each server with `Ctrl+C` after client completion to trigger received-data validation; that final output must still be captured.

## 5. CPU baseline and comparison

`perftest` uses CPU buffers and CPU-posted requests. Run the server first; `-x 3` matches the GID used by GPU latency and `put_bw`, and `--report_gbits` selects Gbit/s:

```bash
# node0
ib_write_lat -d rocep65s0 -x 3
# node1
ib_write_lat -d rocep65s0 -x 3 10.10.1.1
# node0
ib_write_bw -d rocep65s0 -x 3 --report_gbits
# node1
ib_write_bw -d rocep65s0 -x 3 --report_gbits 10.10.1.1
```

| Test | Message size | Iterations | Re-run observations |
| --- | ---: | ---: | --- |
| `ib_write_lat` (node1 client) | 2 B | 1000 | Average **2.50 µs** |
| `ib_write_bw` (both hosts) | 65,536 B (64 KiB) | 5000 | Average **109.56 Gbps** |

`perftest` warned that CPU frequency was not stable at its maximum, adding a possible source of timing variation.

The lab-required comparison uses client latency and average bandwidth at **65,536 B (64 KiB; the handout's 64 KB row)**:

| Measurement | GPUNetIO (GPU buffers, `-p 1` CPU proxy) | `perftest` (CPU buffers, CPU-initiated) |
| --- | ---: | ---: |
| One-way latency, smallest tested message | **8.39 µs (1 B; node1)** | **2.50 µs (2 B; node1)** |
| Average bandwidth at **64 KiB** | **46.755070 Gbps** (`put_bw`, GID 3) | **109.56 Gbps** (`ib_write_bw`, GID 3) |

| Detail | GPUNetIO (`-p 1`) | CPU `perftest` |
| --- | --- | --- |
| RDMA source / destination | GPU memory | Host memory |
| RDMA work constructed / posted | CUDA kernel via GPUNetIO | CPU process |
| NIC doorbell notification | CPU proxy helper thread | CPU thread |
| Transfer mechanism | NIC DMA between registered GPU buffers | NIC DMA between registered CPU buffers |

At 64 KiB, CPU `perftest` achieved **2.34×** the throughput of GPU `put_bw`. The latency payloads differ (1 B versus 2 B), and the applications differ in buffer location, proxy handling, CUDA execution, batching and completion mechanisms. These measurements do not isolate GPU overhead or establish a general CPU/GPU performance ranking.

A CPU-staged receive pipeline needs a host-to-GPU copy and synchronization before GPU inference. Compatible GPUNetIO UDP reception could remove that staging copy, but our UDP sample failed. The supplied Part 7 logs contain no copy timing; quantitative overhead analysis requires the **Part 5 pinned-memory CPU→GPU measurement**.
