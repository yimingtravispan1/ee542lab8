import csv
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.models import ResNet18_Weights

from trt_runner import TensorRTRunner


ROOT = Path(__file__).resolve().parents[1]
SHAPE = (16, 224, 224, 3)
WARMUP = 30
REPEATS = 200


def load_received(mode):
    path = ROOT / "data" / f"frames_rx_{mode}.bin"
    raw = path.read_bytes()

    expected_size = int(np.prod(SHAPE))
    if len(raw) != expected_size:
        raise ValueError(
            f"{path.name}: expected {expected_size} bytes, got {len(raw)}"
        )

    return np.frombuffer(raw, dtype=np.uint8).reshape(SHAPE).copy()


def prepare_buffers(images):
    host = torch.empty(images.shape, dtype=torch.uint8, pin_memory=True)
    host.copy_(torch.from_numpy(np.ascontiguousarray(images)))

    device = torch.empty(images.shape, dtype=torch.uint8, device="cuda")

    if not host.is_pinned():
        raise RuntimeError("Host buffer is not pinned")

    return host, device


def measure_copy_ms(host, device, stream):
    for _ in range(WARMUP):
        device.copy_(host, non_blocking=True)
    stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(stream)
    for _ in range(REPEATS):
        device.copy_(host, non_blocking=True)
    end.record(stream)
    end.synchronize()

    return start.elapsed_time(end) / REPEATS


def classify(images, runner, preprocess, stream):
    host, device = prepare_buffers(images)
    device.copy_(host, non_blocking=True)
    nchw = device.permute(0, 3, 1, 2).contiguous()
    normalized = preprocess(nchw)
    inp = normalized.to(dtype=runner.input_dtype).contiguous()

    if not inp.is_cuda or tuple(inp.shape) != (16, 3, 224, 224):
        raise RuntimeError("Unexpected preprocessed input")

    runner.bind(inp)
    output = runner.run(stream)
    stream.synchronize()

    logits = output.float().cpu()
    if not torch.isfinite(logits).all().item():
        raise RuntimeError("Non-finite logits")

    return logits


@torch.inference_mode()
def main():
    original = np.load(ROOT / "data" / "original_frames.npy")
    if original.shape != SHAPE or original.dtype != np.uint8:
        raise ValueError("Unexpected original image shape or dtype")

    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    categories = weights.meta["categories"]

    runner = TensorRTRunner(
        ROOT / "models" / "resnet18_fp16.engine"
    )
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    classification_rows = []
    copy_rows = []
    summaries = []
    all_matched = True

    with torch.cuda.stream(stream):
        original_logits = classify(
            original, runner, preprocess, stream
        )
        original_classes = original_logits.argmax(dim=1)

        for mode in ("ordered", "shuffled", "duplicates"):
            received = load_received(mode)

            received_logits = classify(
                received, runner, preprocess, stream
            )
            received_classes = received_logits.argmax(dim=1)

            matches = received_classes == original_classes
            matched = int(matches.sum().item())
            max_error = (
                original_logits - received_logits
            ).abs().max().item()

            summaries.append({
                "mode": mode,
                "matched_frames": matched,
                "total_frames": len(original),
                "max_abs_logit_difference": max_error,
            })
            all_matched = all_matched and matched == len(original)

            print(f"\n=== {mode} ===")
            for i in range(len(original)):
                original_id = int(original_classes[i])
                received_id = int(received_classes[i])
                match = bool(matches[i])

                classification_rows.append({
                    "mode": mode,
                    "frame_id": i,
                    "original_class_id": original_id,
                    "received_class_id": received_id,
                    "original_label": categories[original_id],
                    "received_label": categories[received_id],
                    "match": match,
                })

                print(
                    f"Frame {i:02d}: "
                    f"original={original_id}, received={received_id}, "
                    f"{'PASS' if match else 'FAIL'}"
                )

            print(f"Class agreement: {matched}/{len(original)}")
            print(f"Max logit difference: {max_error:.8f}")
        received = load_received("shuffled")

        print("\n=== Pinned-memory CPU -> GPU copy ===")
        for batch in (1, 8, 16):
            host, device = prepare_buffers(received[:batch])
            elapsed_ms = measure_copy_ms(host, device, stream)
            num_bytes = host.numel() * host.element_size()
            gb_per_s = num_bytes / (elapsed_ms / 1000) / 1e9

            copy_rows.append({
                "batch": batch,
                "bytes": num_bytes,
                "copy_ms": elapsed_ms,
                "amortized_ms_per_image": elapsed_ms / batch,
                "effective_GB_s": gb_per_s,
            })

            print(
                f"Batch {batch:2d}: {num_bytes} bytes | "
                f"{elapsed_ms:.6f} ms | "
                f"{gb_per_s:.3f} GB/s"
            )

    for filename, rows in (
        ("part5_classification.csv", classification_rows),
        ("part5_copy_timing.csv", copy_rows),
    ):
        with (ROOT / "results" / filename).open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "engine": "resnet18_fp16.engine",
        "preprocessing": str(preprocess),
        "preprocessing_device": "cuda",
        "classification_batch": 16,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "copy_timing": "CUDA events around repeated pinned H2D copies",
        "excluded_from_copy_timing": [
            "file reading",
            "pinned-memory allocation and CPU staging",
            "GPU allocation",
            "preprocessing",
            "inference",
        ],
        "classification": summaries,
        "copy_results": copy_rows,
        "all_classes_match": all_matched,
    }

    (ROOT / "results" / "part5_summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    if not all_matched:
        raise SystemExit("FAIL: some predicted classes differ")

    print("\nAll received images have the same classes as their originals.")
    print("Part 5 results saved in results/.")


if __name__ == "__main__":
    main()