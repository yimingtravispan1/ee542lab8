import sys
import platform

import torch
import torchvision
import tensorrt as trt
import onnx
import onnxscript
import numpy as np
import scapy


def main():
    print("=== Python environment ===")
    print("Python:", sys.version.split()[0])
    print("Executable:", sys.executable)
    print("Platform:", platform.platform())

    print("\n=== Package versions ===")
    print("PyTorch:", torch.__version__)
    print("torchvision:", torchvision.__version__)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    print("TensorRT:", trt.__version__)
    print("ONNX:", onnx.__version__)
    print("ONNX Script:", onnxscript.__version__)
    print("NumPy:", np.__version__)
    print("Scapy:", scapy.__version__)

    print("\n=== GPU computation ===")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access the GPU")

    device = torch.cuda.get_device_properties(0)
    print("GPU:", device.name)
    print("GPU memory (GiB):", round(device.total_memory / 1024**3, 2))
    print("Compute capability:", torch.cuda.get_device_capability(0))

    x = torch.arange(4, dtype=torch.float32, device="cuda")
    result = x * 2 + 1
    torch.cuda.synchronize()

    expected = torch.tensor([1, 3, 5, 7], dtype=torch.float32)
    torch.testing.assert_close(result.cpu(), expected)
    print("GPU result:", result.cpu().tolist())
    print("PyTorch GPU computation: PASS")

    print("\n=== TensorRT initialization ===")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if builder is None:
        raise RuntimeError("TensorRT builder initialization failed")

    network = builder.create_network()
    if network is None:
        raise RuntimeError("TensorRT network creation failed")

    print("TensorRT builder and network: PASS")
    print("\nAll Python environment checks passed.")


if __name__ == "__main__":
    main()