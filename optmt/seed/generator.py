"""Seed graph generator wrapping NNSmith's model_gen API."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Type

import numpy as np

from nnsmith.abstract.dtype import DTYPE_GEN_FLOATS, DType
from nnsmith.abstract.op import AbsOpBase, Constant, Input
from nnsmith.abstract.tensor import AbsTensor
from nnsmith.gir import GraphIR
from nnsmith.graph_gen import model_gen
from nnsmith.materialize.onnx import ONNXModelCPU

logger = logging.getLogger(__name__)


# Default operator set: core float ops suitable for metamorphic testing
_CORE_OPSET: Optional[List[Type[AbsOpBase]]] = None


def _get_core_opset() -> List[Type[AbsOpBase]]:
    """Lazily load the core operator set from NNSmith materialize registry."""
    global _CORE_OPSET
    if _CORE_OPSET is not None:
        return _CORE_OPSET

    from nnsmith.materialize.torch import TorchModelCPU

    _CORE_OPSET = [
        op for op in TorchModelCPU.operators()
        if op.in_dtypes is not None and op.out_dtypes is not None
    ]
    return _CORE_OPSET


@dataclass
class SeedConfig:
    max_nodes: int = 10
    timeout_ms: int = 10000
    seed: Optional[int] = None
    rank_choices: Optional[List[int]] = None
    dtype_choices: Optional[List[DType]] = None


class SeedGenerator:
    """Generate random seed graphs using NNSmith's symbolic graph generation."""

    def __init__(
        self,
        opset: Optional[List[Type[AbsOpBase]]] = None,
        config: Optional[SeedConfig] = None,
    ):
        self.opset = opset or _get_core_opset()
        self.config = config or SeedConfig()

    def generate(self) -> Tuple[GraphIR, ONNXModelCPU]:
        """Generate a single seed graph and its ONNX model.

        Returns:
            (ir, model): Deep-copied GraphIR and materialized ONNXModelCPU
                          with refined (FP-safe) weights.
        """
        gen = model_gen(
            opset=self.opset,
            method="symbolic",
            max_nodes=self.config.max_nodes,
            seed=self.config.seed,
            timeout_ms=self.config.timeout_ms,
        )
        ir = gen.make_concrete()

        model = ONNXModelCPU.from_gir(ir)
        model.refine_weights()

        return deepcopy(ir), model

    def generate_batch(self, n: int) -> List[Tuple[GraphIR, ONNXModelCPU]]:
        """Generate n seed graphs, skipping failures."""
        results = []
        for i in range(n):
            try:
                results.append(self.generate())
            except Exception as e:
                logger.warning("Seed generation %d failed: %s", i, e)
        return results


def make_random_input(
    input_like: Dict[str, AbsTensor],
    low: float = 1.0,
    high: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """Create random input tensors matching the model's input spec."""
    if rng is None:
        rng = np.random.default_rng()

    inputs = {}
    for name, atensor in input_like.items():
        shape = tuple(atensor.shape)
        dtype_str = str(atensor.dtype).replace("DType.", "").lower()
        # Map NNSmith DType names to numpy dtypes
        np_dtype = {
            "float32": np.float32,
            "float64": np.float64,
            "float16": np.float16,
            "int32": np.int32,
            "int64": np.int64,
            "bool": np.bool_,
        }.get(dtype_str, np.float32)

        if np.issubdtype(np_dtype, np.floating):
            inputs[name] = rng.uniform(low, high, size=shape).astype(np_dtype)
        elif np.issubdtype(np_dtype, np.integer):
            inputs[name] = rng.integers(int(low), int(high) + 1, size=shape).astype(
                np_dtype
            )
        else:
            inputs[name] = rng.choice([True, False], size=shape)

    return inputs
