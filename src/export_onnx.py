from pathlib import Path

import onnx
import torch
from torchvision.models import resnet18, ResNet18_Weights


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"


def check_onnx(path, expected_dtype):
    onnx.checker.check_model(str(path))

    model = onnx.load(str(path), load_external_data=False)
    inp = model.graph.input[0]
    out = model.graph.output[0]

    def shape_of(value):
        return [
            dim.dim_param if dim.dim_param else dim.dim_value
            for dim in value.type.tensor_type.shape.dim
        ]

    print("Input:", inp.name, shape_of(inp))
    print("Output:", out.name, shape_of(out))

    assert inp.name == "input"
    assert out.name == "logits"
    assert inp.type.tensor_type.elem_type == expected_dtype
    assert out.type.tensor_type.elem_type == expected_dtype
    assert inp.type.tensor_type.shape.dim[0].dim_param
    assert out.type.tensor_type.shape.dim[0].dim_param

    assert shape_of(inp)[1:] == [3, 224, 224]
    assert shape_of(out)[1:] == [1000]

    print("ONNX structure, dtype and dynamic batch: PASS")


def export_model(precision):
    dtype = torch.float32 if precision == "fp32" else torch.float16
    onnx_dtype = (
        onnx.TensorProto.FLOAT
        if precision == "fp32"
        else onnx.TensorProto.FLOAT16
    )

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = model.eval().to(device="cuda", dtype=dtype)
    x = torch.randn(2, 3, 224, 224, device="cuda", dtype=dtype)
    batch_dim = torch.export.Dim("batch", min=1, max=32)
    output_path = MODEL_DIR / f"resnet18_{precision}.onnx"

    print(f"\nExporting {precision}: {output_path}", flush=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (x,),
            str(output_path),
            input_names=["input"],
            output_names=["logits"],
            opset_version=18,
            dynamo=True,
            dynamic_shapes=({0: batch_dim},),
            external_data=True,
        )

    check_onnx(output_path, onnx_dtype)
    print(f"{precision} export completed.", flush=True)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.hub.set_dir(str(MODEL_DIR / "torch_cache"))
    torch.manual_seed(542)

    export_model("fp32")
    export_model("fp16")

    print("\nBoth ONNX exports completed successfully.")


if __name__ == "__main__":
    main()