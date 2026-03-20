"""Optimization Attribution: identify which optimization level / passes cause a bug."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from optmt.backend.tvm_relax import TVMRelax
from optmt.oracle.differential import BugReport, DifferentialOracle
from optmt.probe.pass_probe import PassProbe

logger = logging.getLogger(__name__)


@dataclass
class AttributionResult:
    """Result of optimization-level ablation."""

    # Which opt_levels trigger the bug (maps opt_level -> BugReport or None)
    opt_level_results: Dict[int, Optional[BugReport]] = field(default_factory=dict)
    # Passes that fired at the failing opt_level
    fired_passes: List[str] = field(default_factory=list)
    # Whether the bug is optimization-specific (fails at high opt, passes at low opt)
    is_optimization_bug: bool = False
    # Minimum opt_level that triggers the bug
    min_failing_opt_level: Optional[int] = None

    @property
    def summary(self) -> str:
        failing = [k for k, v in self.opt_level_results.items() if v is not None]
        passing = [k for k, v in self.opt_level_results.items() if v is None]
        lines = []
        if self.is_optimization_bug:
            lines.append(
                f"Optimization bug: fails at opt_level={failing}, "
                f"passes at opt_level={passing}"
            )
            lines.append(f"Minimum failing opt_level: {self.min_failing_opt_level}")
        else:
            lines.append("Not clearly optimization-specific (fails at all tested levels)")
        if self.fired_passes:
            lines.append(f"Fired passes at failure: {', '.join(self.fired_passes[:10])}")
        return "\n".join(lines)


def attribute_bug(
    onnx_proto,
    inputs: Dict[str, np.ndarray],
    shape_dict: Optional[Dict[str, List[int]]] = None,
    target: str = "llvm",
    opt_levels: Optional[List[int]] = None,
) -> AttributionResult:
    """Run opt-level ablation to determine if a bug is optimization-specific.

    Tests the same model at multiple optimization levels. If the bug only
    appears at higher levels, it is likely caused by an optimization pass.

    Args:
        onnx_proto: ONNX model proto that triggers the bug
        inputs: Input tensors
        shape_dict: Optional shape dict
        target: TVM target string
        opt_levels: Optimization levels to test (default: [0, 1, 2, 3])

    Returns:
        AttributionResult with per-level findings.
    """
    if opt_levels is None:
        opt_levels = [0, 1, 2, 3]

    result = AttributionResult()

    for opt_level in sorted(opt_levels):
        # pipeline=None disables the default Relax pipeline at opt_level=0
        pipeline = "default" if opt_level > 0 else None
        backend = TVMRelax(target=target, opt_level=opt_level, pipeline=pipeline)
        oracle = DifferentialOracle(backend=backend)

        bug = oracle.compare(onnx_proto, inputs, shape_dict=shape_dict)
        result.opt_level_results[opt_level] = bug

        if bug is not None:
            logger.info(
                "opt_level=%d: BUG (%s, %s)",
                opt_level, bug.symptom.value, bug.severity.value,
            )
        else:
            logger.info("opt_level=%d: PASS", opt_level)

    # Determine if optimization-specific
    failing_levels = sorted(k for k, v in result.opt_level_results.items() if v is not None)
    passing_levels = sorted(k for k, v in result.opt_level_results.items() if v is None)

    if failing_levels and passing_levels:
        result.is_optimization_bug = True
        result.min_failing_opt_level = min(failing_levels)

    # Collect fired passes at the highest failing opt_level
    if failing_levels:
        highest_fail = max(failing_levels)
        probe = PassProbe()
        try:
            backend = TVMRelax(target=target, opt_level=highest_fail)
            backend.compile(onnx_proto, shape_dict, instruments=[probe])
            result.fired_passes = probe.get_fired_passes()
        except Exception:
            # Compilation may crash — that's expected for crash bugs
            result.fired_passes = probe.get_fired_passes()

    return result
