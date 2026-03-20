"""Differential Oracle: compare outputs between reference and SUT backends."""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from optmt.backend.tvm_relax import TVMRelax
from optmt.probe.pass_probe import PassProbe, PassRecord

logger = logging.getLogger(__name__)


class Severity(Enum):
    P0 = "P0"  # Crash, Segfault, Compilation Error
    P1 = "P1"  # Wrong Result, Timeout
    P2 = "P2"  # Precision Anomaly


class Symptom(Enum):
    CRASH = "crash"
    COMPILE_ERROR = "compile_error"
    WRONG_RESULT = "wrong_result"
    PRECISION_ANOMALY = "precision_anomaly"
    TIMEOUT = "timeout"


@dataclass
class BugReport:
    """Report of a detected inconsistency."""

    symptom: Symptom
    severity: Severity
    max_abs_diff: Optional[float] = None
    details: str = ""
    pass_records: List[PassRecord] = field(default_factory=list)
    fired_passes: List[str] = field(default_factory=list)
    error_log: str = ""


class DifferentialOracle:
    """Compare model outputs: same model through ORT (reference) vs Relax (SUT).

    For compiler bug detection, we run the SAME model (same weights, same inputs)
    through two backends. Any output difference indicates a compiler bug.

    Testing flow:
    1. Mutated model → ORT → reference output
    2. Mutated model → Relax (with PassProbe) → test output
    3. If reference ≠ test → BugReport
    """

    def __init__(
        self,
        backend: Optional[TVMRelax] = None,
        atol: float = 1e-5,
        rtol: float = 1e-3,
        wrong_result_threshold: float = 1e-4,
    ):
        self.backend = backend or TVMRelax()
        self.atol = atol
        self.rtol = rtol
        self.wrong_result_threshold = wrong_result_threshold

    def get_reference_output(
        self, onnx_proto, inputs: Dict[str, np.ndarray]
    ) -> List[np.ndarray]:
        """Run model through ONNXRuntime as reference oracle."""
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_proto.SerializeToString())
        input_names = [inp.name for inp in sess.get_inputs()]
        feed = {name: inputs[name] for name in input_names if name in inputs}
        return sess.run(None, feed)

    def compare(
        self,
        onnx_proto,
        inputs: Dict[str, np.ndarray],
        shape_dict: Optional[Dict[str, List[int]]] = None,
    ) -> Optional[BugReport]:
        """Differential test: same model through ORT vs Relax.

        Args:
            onnx_proto: ONNX model proto to test
            inputs: Input tensors
            shape_dict: Shape dict for the model

        Returns:
            BugReport if inconsistency detected, None otherwise.
        """
        # 1. Get reference output from ORT
        try:
            ref_outputs = self.get_reference_output(onnx_proto, inputs)
        except Exception as e:
            logger.warning("ORT reference failed: %s", e)
            return None  # Cannot test without reference

        # 2. Compile and run through Relax with PassProbe
        probe = PassProbe()
        try:
            relax_outputs = self.backend.compile_and_run(
                onnx_proto, inputs, shape_dict, instruments=[probe]
            )
        except Exception as e:
            error_log = traceback.format_exc()
            symptom = Symptom.COMPILE_ERROR
            if "segfault" in str(e).lower() or "signal" in str(e).lower():
                symptom = Symptom.CRASH
            return BugReport(
                symptom=symptom,
                severity=Severity.P0,
                details=str(e),
                pass_records=probe.records,
                fired_passes=probe.get_fired_passes(),
                error_log=error_log,
            )

        # 3. Compare outputs
        return self._compare_outputs(ref_outputs, relax_outputs, probe)

    def compare_metamorphic(
        self,
        original_onnx,
        mutated_onnx,
        original_inputs: Dict[str, np.ndarray],
        mutated_inputs: Dict[str, np.ndarray],
        mutated_shape_dict: Optional[Dict[str, List[int]]] = None,
    ) -> Optional[BugReport]:
        """Full metamorphic differential test.

        Testing flow:
        1. G → ORT → reference output
        2. G_p → ORT → validate RPC equivalence (G ≈ G_p on trusted backend)
        3. G_p → Relax (with PassProbe) → compare with ORT output for G_p
        4. If (2) fails → RPC violation (framework issue, not a compiler bug)
        5. If (3) fails → compiler bug

        Args:
            original_onnx: Original (unmutated) ONNX model
            mutated_onnx: Mutated ONNX model (with RPC-injected pattern)
            original_inputs: Inputs for the original model
            mutated_inputs: Inputs for the mutated model
            mutated_shape_dict: Shape dict for mutated model

        Returns:
            BugReport if compiler bug detected, None otherwise.
        """
        # 1. G → ORT
        try:
            orig_ref = self.get_reference_output(original_onnx, original_inputs)
        except Exception as e:
            logger.warning("Original ORT reference failed: %s", e)
            return None

        # 2. G_p → ORT (validate metamorphic relation)
        try:
            mut_ref = self.get_reference_output(mutated_onnx, mutated_inputs)
        except Exception as e:
            logger.warning("Mutated ORT reference failed: %s", e)
            return None

        rpc_valid = self._check_equivalence(orig_ref, mut_ref)
        if not rpc_valid:
            logger.warning(
                "RPC equivalence violation: G and G_p differ on trusted backend (ORT). "
                "This indicates a framework issue, not a compiler bug."
            )
            return None  # RPC failed, not a compiler bug

        # 3. G_p → Relax
        probe = PassProbe()
        try:
            relax_outputs = self.backend.compile_and_run(
                mutated_onnx, mutated_inputs, mutated_shape_dict, instruments=[probe]
            )
        except Exception as e:
            error_log = traceback.format_exc()
            symptom = Symptom.COMPILE_ERROR
            if "segfault" in str(e).lower() or "signal" in str(e).lower():
                symptom = Symptom.CRASH
            return BugReport(
                symptom=symptom,
                severity=Severity.P0,
                details=str(e),
                pass_records=probe.records,
                fired_passes=probe.get_fired_passes(),
                error_log=error_log,
            )

        # 4. Compare G_p on ORT vs Relax
        return self._compare_outputs(mut_ref, relax_outputs, probe)

    def _check_equivalence(
        self, outputs_a: List[np.ndarray], outputs_b: List[np.ndarray],
        equiv_atol: float = 1e-3, equiv_rtol: float = 1e-2,
    ) -> bool:
        """Check if two output sets are approximately equal (for RPC validation)."""
        if len(outputs_a) != len(outputs_b):
            return False
        for a, b in zip(outputs_a, outputs_b):
            if a.shape != b.shape:
                return False
            try:
                np.testing.assert_allclose(a, b, atol=equiv_atol, rtol=equiv_rtol)
            except AssertionError:
                return False
            except Exception:
                return False
        return True

    def _compare_outputs(
        self,
        ref_outputs: List[np.ndarray],
        test_outputs: List[np.ndarray],
        probe: PassProbe,
    ) -> Optional[BugReport]:
        """Compare reference and test outputs element-wise."""
        if len(ref_outputs) != len(test_outputs):
            return BugReport(
                symptom=Symptom.WRONG_RESULT,
                severity=Severity.P1,
                details=f"Output count mismatch: ref={len(ref_outputs)}, test={len(test_outputs)}",
                pass_records=probe.records,
                fired_passes=probe.get_fired_passes(),
            )

        max_diff = 0.0
        for i, (ref, test) in enumerate(zip(ref_outputs, test_outputs)):
            if ref.shape != test.shape:
                return BugReport(
                    symptom=Symptom.WRONG_RESULT,
                    severity=Severity.P1,
                    details=f"Shape mismatch at output {i}: ref={ref.shape}, test={test.shape}",
                    pass_records=probe.records,
                    fired_passes=probe.get_fired_passes(),
                )

            ref_f = ref.astype(np.float64)
            test_f = test.astype(np.float64)

            # NaN / Inf mismatch detection
            ref_nan = np.isnan(ref_f)
            test_nan = np.isnan(test_f)
            if not np.array_equal(ref_nan, test_nan):
                return BugReport(
                    symptom=Symptom.WRONG_RESULT,
                    severity=Severity.P1,
                    details=f"NaN mismatch at output {i}: ref has {ref_nan.sum()} NaNs, test has {test_nan.sum()} NaNs",
                    pass_records=probe.records,
                    fired_passes=probe.get_fired_passes(),
                )

            ref_inf = np.isinf(ref_f)
            test_inf = np.isinf(test_f)
            if not np.array_equal(ref_inf, test_inf):
                return BugReport(
                    symptom=Symptom.WRONG_RESULT,
                    severity=Severity.P1,
                    details=f"Inf mismatch at output {i}: ref has {ref_inf.sum()} Infs, test has {test_inf.sum()} Infs",
                    pass_records=probe.records,
                    fired_passes=probe.get_fired_passes(),
                )

            # Compute combined atol + rtol difference on finite elements
            finite_mask = np.isfinite(ref_f) & np.isfinite(test_f)
            if not finite_mask.any():
                continue

            abs_diff_arr = np.abs(ref_f[finite_mask] - test_f[finite_mask])
            rel_scale = self.rtol * np.abs(ref_f[finite_mask])
            # Element passes if abs_diff <= atol + rtol * |ref|
            exceeds = abs_diff_arr > (self.atol + rel_scale)
            if exceeds.any():
                abs_diff = float(np.max(abs_diff_arr[exceeds]))
                max_diff = max(max_diff, abs_diff)

        if max_diff > self.wrong_result_threshold:
            return BugReport(
                symptom=Symptom.WRONG_RESULT,
                severity=Severity.P1,
                max_abs_diff=max_diff,
                details=f"Max absolute difference: {max_diff}",
                pass_records=probe.records,
                fired_passes=probe.get_fired_passes(),
            )
        elif max_diff > self.atol:
            return BugReport(
                symptom=Symptom.PRECISION_ANOMALY,
                severity=Severity.P2,
                max_abs_diff=max_diff,
                details=f"Precision anomaly: max_abs_diff={max_diff}",
                pass_records=probe.records,
                fired_passes=probe.get_fired_passes(),
            )

        return None
