import csv
import json
from pathlib import Path

import torch
import tensorrt as trt
from torchvision.models import resnet18, ResNet18_Weights

from trt_runner import TensorRTRunner


ROOT = Path(__file__).resolve().parents[1]
WARMUP = 50
REPEATS = 200


def measure_ms(fn, stream):
    for _ in range(WARMUP):
        fn()
    stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream)
    for _ in range(REPEATS):
        fn()
    end.record(stream)
    end.synchronize()

    return start.elapsed_time(end) / REPEATS


def compare_logits(reference, output):
    output = output.float()

    if not torch.isfinite(reference).all().item():
        raise RuntimeError("Non-finite PyTorch output")
    if not torch.isfinite(output).all().item():
        raise RuntimeError("Non-finite TensorRT output")

    max_error = (reference - output).abs().max().item()
    agreement = (
        reference.argmax(dim=1) == output.argmax(dim=1)
    ).float().mean().item()

    return max_error, agreement * 100


@torch.inference_mode()
def main():
    torch.manual_seed(542)
    torch.hub.set_dir(str(ROOT / "models" / "torch_cache"))
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.benchmark = False

    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval().cuda()

    runners = {
        "TensorRT FP32": TensorRTRunner(
            ROOT / "models" / "resnet18_fp32.engine"
        ),
        "TensorRT FP16": TensorRTRunner(
            ROOT / "models" / "resnet18_fp16.engine"
        ),
    }

    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    rows = []

    with torch.cuda.stream(stream):
        for batch in (1, 8, 32):
            print(f"\n=== Batch {batch} ===", flush=True)

            x = torch.randn(batch, 3, 224, 224, device="cuda")
            reference = model(x)
            stream.synchronize()

            pytorch_ms = measure_ms(lambda: model(x), stream)

            rows.append({
                "implementation": "PyTorch FP32",
                "batch": batch,
                "latency_ms": pytorch_ms,
                "throughput_images_s": batch * 1000 / pytorch_ms,
                "speedup": 1.0,
                "max_abs_error": 0.0,
                "top1_agreement_pct": 100.0,
            })

            print(f"PyTorch FP32: {pytorch_ms:.4f} ms", flush=True)

            for name, runner in runners.items():
                inp = x.to(dtype=runner.input_dtype).contiguous()
                runner.bind(inp)

                output = runner.run(stream)
                stream.synchronize()

                max_error, agreement = compare_logits(reference, output)

                latency_ms = measure_ms(
                    lambda: runner.run(stream),
                    stream,
                )
                speedup = pytorch_ms / latency_ms

                rows.append({
                    "implementation": name,
                    "batch": batch,
                    "latency_ms": latency_ms,
                    "throughput_images_s": batch * 1000 / latency_ms,
                    "speedup": speedup,
                    "max_abs_error": max_error,
                    "top1_agreement_pct": agreement,
                })

                print(
                    f"{name}: {latency_ms:.4f} ms | "
                    f"speedup={speedup:.2f}x | "
                    f"max_error={max_error:.6f} | "
                    f"top1={agreement:.2f}%",
                    flush=True,
                )

    csv_path = ROOT / "results" / "part2_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "gpu": torch.cuda.get_device_name(0),
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "tensorrt": trt.__version__,
        "seed": 542,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "timing": "CUDA events around repeated inference on a non-default stream",
        "input": "Random normal CUDA tensors; FP16 uses rounded copies",
        "pytorch_precision": "IEEE FP32",
        "tensorrt_precision": "As configured in saved engines",
        "correctness_samples": "One batch per batch size: 1, 8, 32",
        "results": rows,
    }
    json_path = ROOT / "results" / "part2_benchmark.json"
    json_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()