# EE542 Lab 8 Report

## Overview

We implemented and tested a pipeline that converts a pretrained CNN to TensorRT, represents images as UDP packets, reconstructs the images using DPDK, and classifies the reconstructed images on the GPU. We also built and tested NVIDIA DOCA GPUNetIO samples to examine the requirements for direct network access to GPU memory.

Parts 1–5 were performed on a local NVIDIA GPU under WSL2. Part 6 was also tested locally using Docker Desktop instead of the cloud VM specified in the handout. This environment difference is reported explicitly. The Part 6 results demonstrate successful compilation and identify runtime device limitations; they do not demonstrate successful GPU network communication.

Part 7 CloudLab experiments are outside the scope of this report.

## Part 1: Environment Setup

The project was developed in VS Code, with Python dependencies installed in a project-specific virtual environment.

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| GPU memory reported by nvidia-smi | 12,227 MiB |
| Compute capability | 12.0 |
| Operating system | Ubuntu 24.04 under WSL2 |
| NVIDIA kernel-mode driver version | 617.42 |
| NVIDIA-SMI version | 615.78.05 |
| CUDA UMD version reported by nvidia-smi | 13.4 |
| Host CUDA Toolkit | 13.1 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| TensorRT | 11.4.0.106 |
| DPDK | 23.11.4 |
| Scapy | 2.8.0 |
| DOCA | 3.5.0098 |
| CUDA Toolkit inside the DOCA container | 13.0 |

The host toolkit, PyTorch CUDA build, and container toolkit are separate software components, so their version numbers differ. The CUDA UMD version reported by `nvidia-smi` is not the installed toolkit version.

The environment supported GPU inference, TensorRT engine construction, DPDK PCAP reception, and compilation of the DOCA samples.

Evidence: [Python dependencies](requirements.txt), [GPU and driver information](logs/nvidia-smi.txt), and [CUDA compiler information](logs/nvcc.txt).

## Part 2: TensorRT Inference from GPU Memory

### Implementation

We used pretrained torchvision ResNet-18 and exported separate FP32 and FP16 ONNX models with a dynamic batch dimension. TensorRT engines used an optimization profile with batch sizes of 1, 8, and 32 as the minimum, optimal, and maximum values.

The inference runner allocated input and output tensors in GPU memory and passed their device pointers to TensorRT using `set_tensor_address`. Inference was submitted through `execute_async_v3` on a CUDA stream. Tensor data types matched the engine requirements, and execution was synchronized before reading results.

The FP16 engine was built from the FP16 ONNX model, following TensorRT 11's strongly typed workflow.

### Benchmark Results

Measurements used 50 warm-up iterations followed by 200 timed iterations. CUDA events measured GPU execution time.

| Batch | Implementation | Latency (ms) | Throughput (images/s) | Speedup |
|---:|---|---:|---:|---:|
| 1 | PyTorch FP32 | 1.9484 | 513.2 | 1.00× |
| 1 | TensorRT FP32 | 0.8331 | 1,200.4 | 2.34× |
| 1 | TensorRT FP16 | 0.2937 | 3,404.8 | 6.63× |
| 8 | PyTorch FP32 | 3.4753 | 2,301.9 | 1.00× |
| 8 | TensorRT FP32 | 1.5483 | 5,166.8 | 2.24× |
| 8 | TensorRT FP16 | 0.5280 | 15,152.7 | 6.58× |
| 32 | PyTorch FP32 | 7.6124 | 4,203.7 | 1.00× |
| 32 | TensorRT FP32 | 5.1113 | 6,260.6 | 1.49× |
| 32 | TensorRT FP16 | 1.6769 | 19,082.3 | 4.54× |

Calculations:

- Speedup = PyTorch latency / TensorRT latency.
- Throughput = batch size × 1,000 / latency in milliseconds.

TensorRT FP16 achieved the lowest latency at every tested batch size. Its highest measured throughput was approximately 19,082 images/s at batch 32. Increasing the batch size improved throughput but increased the time required to process each batch.

### Numerical Consistency

Each engine was compared with PyTorch FP32 on the same input.

| Batch | TensorRT FP32 maximum absolute logit error | TensorRT FP16 maximum absolute logit error | FP32 top-1 agreement | FP16 top-1 agreement |
|---:|---:|---:|---:|---:|
| 1 | 0.003281 | 0.014122 | 100% | 100% |
| 8 | 0.004418 | 0.021438 | 100% | 100% |
| 32 | 0.006119 | 0.028236 | 100% | 100% |

FP16 produced larger numerical differences, but the top-1 predictions matched PyTorch for all tested inputs. TensorRT FP32 retained its default configuration, which may permit TF32 operations.

These comparisons used random inputs and establish agreement on those inputs, not classification accuracy on a labeled dataset. The timings measure inference on GPU-resident buffers, excluding packet reception and host-to-device transfers.

Evidence: [benchmark results](results/part2_benchmark.csv).

## Part 3: Images as UDP Packets

We generated 16 synthetic RGB images, each with dimensions 224 × 224. Each image contained:

`224 × 224 × 3 = 150,528 bytes`

Each UDP payload contained a 22-byte application header followed by up to 1,400 bytes of image data.

| Header field | Size | Purpose |
|---|---:|---|
| magic | 4 bytes | Identifies the application packet format |
| frame_id | 4 bytes | Identifies the image |
| seq | 2 bytes | Identifies the segment |
| nseg | 2 bytes | Specifies the total segment count |
| offset | 4 bytes | Specifies the destination offset within the image |
| length | 2 bytes | Specifies the image bytes in this packet |
| frame_bytes | 4 bytes | Specifies the complete image size |

Each image required:

`ceil(150,528 / 1,400) = 108 packets`

The final packet contained 728 image bytes. Therefore, 16 images produced 1,728 packets.

The largest IPv4 packet occupied:

`20 + 8 + 22 + 1,400 = 1,450 bytes`

This includes the IPv4 header, UDP header, application header, and image chunk. It fits within a 1,500-byte IP MTU; Ethernet framing is additional to that MTU.

We saved the original images and generated both ordered and shuffled PCAP files. A separate robustness test added 100 duplicate packets.

Evidence: [packet generator](src/generate_pcap.py), [duplicate generator](src/generate_duplicates.py), and [generation log](logs/generate_pcap.txt).

## Part 4: DPDK Reception and Image Reassembly

### Receiver Design

The C receiver used DPDK's PCAP virtual port to process packets without requiring a physical high-speed NIC.

The receiver validated Ethernet, IPv4, UDP, and application headers, checked the destination port, interpreted network-byte-order fields, and validated chunk boundaries before copying data.

Chunks were placed according to their frame IDs and byte offsets. Segment tracking allowed arbitrary arrival order and prevented duplicate chunks from being counted more than once.

The PCAP port does not provide an explicit end-of-file notification. The receiver therefore used a two-second inactivity timeout, allowing it to process trailing duplicate packets before stopping. This is an offline-test heuristic rather than a general completion rule for live traffic.

### Results

| Test | Packets received | Accepted chunks | Invalid/unrelated packets dropped | Duplicate chunks ignored | Complete frames |
|---|---:|---:|---:|---:|---:|
| Ordered | 1,728 | 1,728 | 0 | 0 | 16/16 |
| Shuffled | 1,728 | 1,728 | 0 | 0 | 16/16 |
| With duplicates | 1,828 | 1,728 | 0 | 100 | 16/16 |

For ordered and shuffled inputs, the reconstructed output contained 2,408,448 bytes and matched the original data byte for byte. The original and reconstructed SHA-256 hashes also matched.

The duplicate test completed all 16 frames while ignoring exactly 100 repeated chunks. Its downstream classification results were also identical to those of the original images.

These tests validate ordered, shuffled, and duplicate handling. They do not establish behavior under every malformed-packet or packet-loss condition.

Evidence: [receiver source](src/dpdk_rx.c), [byte-level verification](results/part4_verification.json), and [duplicate-test log](logs/dpdk_duplicates.txt).

## Part 5: Classifying Reconstructed Images

### GPU Processing

The reconstructed images were staged in pinned CPU memory and copied to the GPU. Preprocessing was performed on the GPU using the transforms associated with the pretrained ResNet-18 weights:

- Conversion from HWC image bytes to channel-first tensors.
- Resizing to 256 and center cropping to 224 × 224.
- Conversion to floating-point values and normalization.
- Conversion to FP16 for TensorRT inference.

Normalization used mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

Original and reconstructed images were evaluated using the same preprocessing and FP16 engine.

| Input mode | Matching top-1 predictions | Maximum absolute logit difference |
|---|---:|---:|
| Ordered | 16/16 | 0 |
| Shuffled | 16/16 | 0 |
| Duplicate packets | 16/16 | 0 |

All reconstructed images produced the same classifications and logits as their originals. Because the images were synthetic, this result demonstrates pipeline consistency rather than semantic classification accuracy.

### Pinned-Memory Copy Timing

Host-to-device transfer timing used CUDA events, 30 warm-up iterations, and 200 measured iterations.

| Batch | Bytes transferred | Average copy time (ms) | Effective bandwidth (GB/s) |
|---:|---:|---:|---:|
| 1 | 150,528 | 0.007420 | 20.287 |
| 8 | 1,204,224 | 0.028415 | 42.380 |
| 16 | 2,408,448 | 0.055784 | 43.175 |

The timing excluded file reading, pinned-memory allocation, CPU staging, GPU allocation, preprocessing, and inference.

Larger batches achieved higher effective bandwidth, consistent with fixed transfer overhead being amortized across more data. Copying all 16 images took approximately 55.8 microseconds.

### Relationship to GPUNetIO

Our implemented path is:

`PCAP → DPDK → CPU image buffers → pinned-memory staging → GPU copy → GPU preprocessing → TensorRT`

A supported GPUNetIO receive path can place incoming packet data directly into GPU memory, eliminating the explicit CPU-to-GPU staging copy. However, GPU-side packet handling, image assembly, synchronization, preprocessing, and inference still require work.

The measured copy time identifies one source of overhead. It does not establish the total performance benefit of GPUNetIO, because a working direct-receive path was not available in the local environment.

Evidence: [classification source](src/classify_frames.py), [classification results](results/part5_classification.csv), and [copy timing](results/part5_copy_timing.csv).

## Part 6: NVIDIA DOCA Container

### Docker GPU Access and DOCA Inspection

We used the following container image:

`nvcr.io/nvidia/doca/doca:devel-cuda13.0.0-3.5.0-devel-host`

Docker successfully exposed the RTX 5070. The container provided CUDA 13.0 and DOCA 3.5.0098.

The installed-library list included `common`, `argp`, `eth`, `flow`, `gpunetio`, `rdma`, `verbs`, `dma`, and other DOCA components. The full list is saved in [part6_doca_libs.txt](logs/part6_doca_libs.txt).

Device enumeration reported:

`No DOCA device was found`

This distinguishes software availability from hardware availability: the libraries were installed, but the environment did not expose a compatible DOCA network device.

### Building the Ethernet Receive Sample

We built `gpunetio_simple_receive`, which is designed to receive UDP packets into GPU memory and process reception using a CUDA kernel.

For the RTX 5070, we added `compute_120,code=sm_120` to both the compilation and linking flags in `meson.build`. The compilation flag had to be added before `add_project_arguments`; adding it afterward initially produced a missing-architecture warning during device linking.

After correcting the ordering, all build steps completed successfully without that warning. The modified configuration is saved as [meson.part6.build](meson.part6.build).

### Resolving CUDA Initialization

The first execution failed during `cudaFree(0)`:

`CUDA initialization failed: no CUDA-capable device is detected`

Inspection showed that the application loaded the compatibility driver library from:

`/usr/local/cuda-13/compat/libcuda.so.1`

After adjusting `LD_LIBRARY_PATH` to remove the compatibility directory, it loaded:

`/usr/lib/x86_64-linux-gnu/libcuda.so.1`

CUDA initialization then succeeded, and the program proceeded to its network-device initialization.

### Ethernet Sample Runtime Result

The negative test used:

```bash
/tmp/build/doca_gpunetio_simple_receive \
  -g 0000:01:00.0 \
  -n 0000:00:00.0 \
  -e 0
```

The GPU address was detected from the system. The NIC address was a placeholder, not a detected DOCA device. The `-e 0` option selected per-thread reception.

The program stopped in `init_doca_device` with:

```text
Matching device not found
Failed to open NIC device based on PCI address
```

This result is consistent with the absence of an exposed DOCA network device. Successful compilation and CUDA initialization were demonstrated, but actual GPU packet reception was not.

Evidence: [successful build](logs/part6_build.txt) and [runtime log after the library fix](logs/part6_sample_run_after_library_fix.txt).

### Open-Source GPUNetIO RDMA Example

We also built the open-source NVIDIA GPUNetIO library and examples:

```bash
make install install_examples \
  PREFIX=/work/gpunetio/install \
  CUDA_ARCH=120
```

The installed library directory was added to `LD_LIBRARY_PATH`, and the latency example was run with logging enabled:

```bash
DOCA_GPUNETIO_LOG=6 \
/work/gpunetio/install/examples/gpunetio_verbs_write_lat \
  -g 0000:01:00.0 \
  -d mlx5_0
```

The name `mlx5_0` was a negative-test placeholder; no RDMA device with that name was detected locally.

The log reported `Use DOCA GPUNetIO open`. GDRCopy could not be loaded, and its usage was disabled. Execution continued until `open_ib_device` failed with:

```text
Failed to get RDMA devices list, ibv_dev_list null
```

The program subsequently failed to create local RDMA resources. The empty `/sys/class/infiniband/` listing was consistent with this result.

The missing GDRCopy library was not the immediate cause of termination. The decisive failure was the inability to obtain an RDMA device list.

The open-source implementation provides an RDMA Verbs GPU data path, not an Ethernet UDP receive path. Successful communication requires suitable hardware and client/server endpoints. No RDMA latency or bandwidth measurement was obtained in this local test.

Evidence: [open-source build log](logs/part6_gpunetio_build.txt) and [execution log](logs/part6_open_gpunetio_run.txt).


## References

- EE542 Laboratory 8 handout.
- [NVIDIA DOCA GPUNetIO installation and setup](https://networking-docs.nvidia.com/doca/archive/3-5-0/gpunetio-installation-and-setup)
- [NVIDIA DOCA GPUNetIO sample guide](https://networking-docs.nvidia.com/doca/archive/3-5-0/gpunetio-sample-guide)
- [NVIDIA open-source GPUNetIO](https://github.com/NVIDIA-DOCA/gpunetio)
