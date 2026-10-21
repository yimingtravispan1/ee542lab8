from pathlib import Path

import torch
import tensorrt as trt


class TensorRTRunner:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        engine_bytes = Path(engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Cannot load engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create execution context")

        dtype_map = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
        }
        self.input_dtype = dtype_map[self.engine.get_tensor_dtype("input")]
        self.output_dtype = dtype_map[self.engine.get_tensor_dtype("logits")]

        self.inp = None
        self.out = None

    def bind(self, inp):
      
        if not inp.is_cuda or not inp.is_contiguous():
            raise ValueError("Input must be a contiguous CUDA tensor")
        if inp.dtype != self.input_dtype:
            raise ValueError(f"Expected {self.input_dtype}, got {inp.dtype}")
        if inp.ndim != 4 or tuple(inp.shape[1:]) != (3, 224, 224):
            raise ValueError(f"Unexpected input shape: {tuple(inp.shape)}")
        if not 1 <= inp.shape[0] <= 32:
            raise ValueError("Batch must be between 1 and 32")

        if not self.context.set_input_shape("input", tuple(inp.shape)):
            raise RuntimeError("Cannot set input shape")

        output_shape = tuple(self.context.get_tensor_shape("logits"))
        if output_shape != (inp.shape[0], 1000):
            raise RuntimeError(f"Unexpected output shape: {output_shape}")

    
        self.inp = inp
        self.out = torch.empty(
            output_shape,
            device=inp.device,
            dtype=self.output_dtype,
        )

        if not self.context.set_tensor_address("input", self.inp.data_ptr()):
            raise RuntimeError("Cannot bind input address")
        if not self.context.set_tensor_address("logits", self.out.data_ptr()):
            raise RuntimeError("Cannot bind output address")

    def run(self, stream):
        if self.inp is None:
            raise RuntimeError("Call bind() first")

        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return self.out