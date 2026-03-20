"""TVM Relax backend: ONNX → Relax IR → VirtualMachine execution."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import tvm
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax import build as relax_build
from tvm.runtime.relax_vm import VirtualMachine

from nnsmith.abstract.tensor import AbsTensor
from nnsmith.materialize.onnx import ONNXModel

logger = logging.getLogger(__name__)


class TVMRelax:
    """TVM Relax backend that compiles ONNX models via the Relax pipeline."""

    def __init__(
        self,
        target: str = "llvm",
        opt_level: int = 3,
        pipeline: Optional[str] = "default",
    ):
        self.target = tvm.target.Target(target)
        self.opt_level = opt_level
        self.pipeline = pipeline

    def get_device(self) -> tvm.runtime.Device:
        dev_keys = self.target.export()["keys"]
        if "cuda" in dev_keys:
            return tvm.cuda()
        if "rocm" in dev_keys:
            return tvm.rocm()
        return tvm.cpu()

    def compile(
        self,
        onnx_model,
        shape_dict: Optional[Dict[str, List[int]]] = None,
        instruments: Optional[List] = None,
    ) -> tvm.runtime.Module:
        """Compile an ONNX model to a Relax executable.

        Args:
            onnx_model: ONNX ModelProto
            shape_dict: Optional input shape overrides
            instruments: Optional list of PassInstrument instances

        Returns:
            Compiled executable for VirtualMachine
        """
        mod = from_onnx(onnx_model, shape_dict=shape_dict)

        pass_ctx_kwargs = {"opt_level": self.opt_level}
        if instruments:
            pass_ctx_kwargs["instruments"] = instruments

        with tvm.transform.PassContext(**pass_ctx_kwargs):
            ex = relax_build(mod, target=self.target, relax_pipeline=self.pipeline)

        return ex

    def run(
        self,
        executable: tvm.runtime.Module,
        inputs: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Run inference using the compiled executable.

        Args:
            executable: Compiled Relax executable
            inputs: Ordered list of input numpy arrays

        Returns:
            List of output numpy arrays
        """
        device = self.get_device()
        vm = VirtualMachine(executable, device)
        tvm_inputs = [tvm.nd.array(v, device) for v in inputs]
        output = vm["main"](*tvm_inputs)
        return _cvt_result(output)

    def compile_and_run(
        self,
        onnx_model,
        inputs: Dict[str, np.ndarray],
        shape_dict: Optional[Dict[str, List[int]]] = None,
        instruments: Optional[List] = None,
    ) -> List[np.ndarray]:
        """Compile and run in one step."""
        ex = self.compile(onnx_model, shape_dict, instruments)
        # Order inputs by ONNX model's declared input order
        ordered_inputs = _order_inputs_by_onnx(onnx_model, inputs)
        return self.run(ex, ordered_inputs)

    def make_backend(self, model: ONNXModel):
        """Create a callable backend closure compatible with NNSmith's interface.

        Args:
            model: ONNXModel instance

        Returns:
            Callable that takes Dict[str, np.ndarray] → Dict[str, np.ndarray]
        """
        onnx_proto = model.native_model
        shape_dict = {
            name: list(aten.shape) for name, aten in model.input_like.items()
        }
        ex = self.compile(onnx_proto, shape_dict)
        device = self.get_device()
        vm = VirtualMachine(ex, device)

        def closure(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            ordered = _order_inputs_by_onnx(onnx_proto, inputs)
            tvm_inputs = [tvm.nd.array(v, device) for v in ordered]
            output = vm["main"](*tvm_inputs)
            results = _cvt_result(output)
            return dict(zip(model.output_like.keys(), results))

        return closure


def _order_inputs_by_onnx(onnx_model, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    """Order input arrays by the ONNX model's declared input signature."""
    ordered = []
    for inp in onnx_model.graph.input:
        name = inp.name
        if name in inputs:
            ordered.append(inputs[name])
    # Fallback: if ONNX graph has no input metadata, use dict order
    if not ordered:
        ordered = list(inputs.values())
    return ordered


def _cvt_result(output: Any) -> List[np.ndarray]:
    """Convert VM output (NDArray, tuple, or list) to list of numpy arrays."""
    if output is None:
        raise RuntimeError("VM returned None output")
    if isinstance(output, tvm.nd.NDArray):
        return [output.numpy()]
    # Handle iterable outputs (list, tuple, tvm.ffi.container.Array, etc.)
    if hasattr(output, "__len__") and not isinstance(output, np.ndarray):
        return [r.numpy() if hasattr(r, "numpy") else np.array(r) for r in output]
    # Try numpy conversion for other TVM types
    if hasattr(output, "numpy"):
        return [output.numpy()]
    return [np.array(output)]
