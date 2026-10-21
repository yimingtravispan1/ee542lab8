from pathlib import Path
import time

import tensorrt as trt


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(precision):
    onnx_path = MODEL_DIR / f"resnet18_{precision}.onnx"
    engine_path = MODEL_DIR / f"resnet18_{precision}.engine"

    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    print(f"\n=== Building {precision} engine ===", flush=True)
    start = time.perf_counter()

    builder = trt.Builder(LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [
            str(parser.get_error(i))
            for i in range(parser.num_errors)
        ]
        raise RuntimeError("ONNX parsing failed:\n" + "\n".join(errors))

    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError("Expected one input and one output")

    inp = network.get_input(0)
    out = network.get_output(0)

    print("Input:", inp.name, inp.shape, inp.dtype)
    print("Output:", out.name, out.shape, out.dtype)

    expected_dtype = trt.float32 if precision == "fp32" else trt.float16
    if inp.dtype != expected_dtype or out.dtype != expected_dtype:
        raise RuntimeError("Unexpected input/output dtype")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        2 * 1024**3,
    )
    profile = builder.create_optimization_profile()
    profile.set_shape(
        inp.name,
        min=(1, 3, 224, 224),
        opt=(8, 3, 224, 224),
        max=(32, 3, 224, 224),
    )

    if not profile:
        raise RuntimeError("Invalid optimization profile")

    print("Configured shapes:", profile.get_shape(inp.name), flush=True)

    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("Failed to add optimization profile")

    print(
        "Batch profile: min=1, opt=8, max=32",
        flush=True,
    )
    print("Building engine; this may take several minutes...", flush=True)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Failed to build {precision} engine")
    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("Engine deserialization failed")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create execution context")

    for batch in (1, 8, 32):
        if not context.set_input_shape("input", (batch, 3, 224, 224)):
            raise RuntimeError(f"Failed to set batch={batch}")

        output_shape = tuple(context.get_tensor_shape("logits"))
        if output_shape != (batch, 1000):
            raise RuntimeError(f"Unexpected output shape: {output_shape}")

        print(f"Batch {batch}: output shape {output_shape} — PASS")

    engine_path.write_bytes(bytes(serialized))

    elapsed = time.perf_counter() - start
    size_mib = engine_path.stat().st_size / 1024**2

    print(f"Saved: {engine_path}")
    print(f"Engine size: {size_mib:.2f} MiB")
    print(f"Build and validation time: {elapsed:.2f} seconds")


def main():
    print("TensorRT version:", trt.__version__)

    build_engine("fp32")
    build_engine("fp16")

    print("\nBoth TensorRT engines built successfully.")


if __name__ == "__main__":
    main()