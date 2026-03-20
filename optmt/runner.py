"""Fuzzing runner: main loop for metamorphic testing.

Supports two mutation backends:
  - Legacy (GraphIR): Mutator → ONNXModelCPU.from_gir()
  - ONNX-Native: OnnxMutator → direct ONNX protobuf mutation
"""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from random import randint
from typing import Dict, List, Optional, Union

import numpy as np

from nnsmith.gir import GraphIR
from nnsmith.materialize.onnx import ONNXModelCPU

from optmt.backend.tvm_relax import TVMRelax
from optmt.mutator import Mutator, MutationRecord
from optmt.oracle.differential import BugReport, DifferentialOracle
from optmt.oracle.attribution import attribute_bug, AttributionResult
from optmt.reducer.delta_debug import DeltaDebugger
from optmt.seed.generator import SeedGenerator, SeedConfig, make_random_input

logger = logging.getLogger(__name__)


class FuzzRunner:
    """Main fuzzing loop: generate -> mutate -> compile -> compare -> reduce -> report."""

    def __init__(
        self,
        seed_gen: Optional[SeedGenerator] = None,
        mutator: Optional[Mutator] = None,
        oracle: Optional[DifferentialOracle] = None,
        output_dir: str = "fuzz_output",
        max_mutations: int = 3,
        enable_reduction: bool = True,
        enable_attribution: bool = True,
        onnx_mutator=None,
    ):
        self.seed_gen = seed_gen or SeedGenerator()
        self.mutator = mutator or Mutator()
        self.oracle = oracle or DifferentialOracle()
        self.output_dir = Path(output_dir)
        self.max_mutations = max_mutations
        self.enable_reduction = enable_reduction
        self.enable_attribution = enable_attribution
        self.onnx_mutator = onnx_mutator  # OnnxMutator instance or None

        # Statistics
        self.stats = {
            "total_seeds": 0,
            "total_mutations": 0,
            "bugs_found": 0,
            "bugs_reduced": 0,
            "errors": 0,
            "silent_skips": 0,
            "start_time": 0,
        }

    def run(self, time_budget_s: float = 300, max_iterations: Optional[int] = None):
        """Run the fuzzing loop.

        Args:
            time_budget_s: Maximum time in seconds.
            max_iterations: Maximum number of iterations (None = unlimited).
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        self.stats["start_time"] = start
        iteration = 0

        logger.info("Starting fuzzing with budget=%ds, output=%s", time_budget_s, self.output_dir)

        while time.time() - start < time_budget_s:
            if max_iterations is not None and iteration >= max_iterations:
                break

            iteration += 1
            try:
                self._fuzz_one(iteration)
            except Exception as e:
                self.stats["errors"] += 1
                logger.warning("Iteration %d error: %s", iteration, e)

        elapsed = time.time() - start
        logger.info(
            "Fuzzing complete: %d iterations, %d bugs found (%d reduced), %d errors in %.1fs",
            iteration, self.stats["bugs_found"], self.stats["bugs_reduced"],
            self.stats["errors"], elapsed,
        )
        self._save_stats()

    def _fuzz_one(self, iteration: int):
        """Execute one fuzzing iteration."""
        if self.onnx_mutator is not None:
            self._fuzz_one_onnx(iteration)
        else:
            self._fuzz_one_legacy(iteration)

    # ── ONNX-Native path ────────────────────────────────────

    def _fuzz_one_onnx(self, iteration: int):
        """ONNX-native mutation path: mutate directly on ONNX protobuf."""
        from optmt.injection.onnx_mutator import OnnxMutationRecord
        from optmt.reducer.delta_debug import OnnxDeltaDebugger

        MAX_SEED_RETRIES = 5
        for attempt in range(MAX_SEED_RETRIES):
            # 1. Generate seed
            ir, model = self.seed_gen.generate()
            self.stats["total_seeds"] += 1

            seed_onnx = model.native_model
            inputs = make_random_input(model.input_like)
            shape_dict = {
                name: list(aten.shape) for name, aten in model.input_like.items()
            }

            # 2. Mutate at ONNX level (same graph inputs, same shapes)
            n_mutations = randint(1, self.max_mutations)
            try:
                mutated_onnx, records = self.onnx_mutator.mutate(
                    seed_onnx, n_mutations=n_mutations
                )
                break  # success
            except Exception:
                if attempt == MAX_SEED_RETRIES - 1:
                    raise
                logger.debug(
                    "Mutation failed, retrying with new seed (%d/%d)",
                    attempt + 1, MAX_SEED_RETRIES,
                )
                continue

        self.stats["total_mutations"] += len(records)

        # 3. Metamorphic test (same inputs for seed & mutant)
        bug = self.oracle.compare_metamorphic(
            seed_onnx, mutated_onnx,
            inputs, inputs,
            mutated_shape_dict=shape_dict,
        )

        # 4. Fallback cross-backend test
        if bug is None:
            bug = self.oracle.compare(
                mutated_onnx, inputs, shape_dict=shape_dict,
            )

        # 5. Bug found — reduce and save
        if bug is not None:
            self.stats["bugs_found"] += 1
            logger.info(
                "Bug #%d found at iteration %d: %s (severity=%s)",
                self.stats["bugs_found"], iteration,
                bug.symptom.value, bug.severity.value,
            )

            reduced_records = records
            if self.enable_reduction and len(records) > 1:
                reduced_records = self._reduce_onnx(seed_onnx, records, inputs, shape_dict)

            self._save_bug_onnx(
                bug, seed_onnx, records, reduced_records,
                inputs, shape_dict, mutated_onnx, iteration,
            )
        else:
            self.stats["silent_skips"] += 1

    def _reduce_onnx(
        self,
        seed_onnx,
        records,
        inputs: Dict[str, np.ndarray],
        shape_dict: Dict[str, List[int]],
    ):
        """ONNX-native delta debugging."""
        from optmt.reducer.delta_debug import OnnxDeltaDebugger

        def reproduces_bug(mutated_onnx) -> Optional[BugReport]:
            try:
                return self.oracle.compare_metamorphic(
                    seed_onnx, mutated_onnx,
                    inputs, inputs,
                    mutated_shape_dict=shape_dict,
                )
            except Exception as e:
                logger.debug("ONNX reduction predicate error: %s", e)
                return None

        debugger = OnnxDeltaDebugger(
            seed_onnx, records, self.onnx_mutator, reproduces_bug,
        )
        try:
            minimized = debugger.minimize()
            if len(minimized) < len(records):
                self.stats["bugs_reduced"] += 1
                logger.info(
                    "Reduced %d -> %d ONNX mutation records",
                    len(records), len(minimized),
                )
            return minimized
        except Exception as e:
            logger.warning("ONNX delta debugging failed: %s", e)
            return records

    def _save_bug_onnx(
        self,
        bug: BugReport,
        seed_onnx,
        records,
        reduced_records,
        inputs: Dict[str, np.ndarray],
        shape_dict,
        mutated_onnx,
        iteration: int,
    ):
        """Save bug report for ONNX-native path."""
        from optmt.injection.onnx_mutator import OnnxMutationRecord

        bug_dir = self.output_dir / f"bug_{self.stats['bugs_found']:04d}_iter{iteration}"
        bug_dir.mkdir(parents=True, exist_ok=True)

        import onnx
        onnx.save(seed_onnx, str(bug_dir / "seed.onnx"))
        onnx.save(mutated_onnx, str(bug_dir / "mutated.onnx"))

        records_data = [r.to_dict() for r in records]
        with open(bug_dir / "mutation_records.json", "w") as f:
            json.dump(records_data, f, indent=2)

        if len(reduced_records) < len(records):
            reduced_data = [r.to_dict() for r in reduced_records]
            with open(bug_dir / "mutation_records_reduced.json", "w") as f:
                json.dump(reduced_data, f, indent=2)

        attribution_data = {}
        if self.enable_attribution:
            try:
                attr_result = attribute_bug(mutated_onnx, inputs, shape_dict)
                attribution_data = {
                    "is_optimization_bug": attr_result.is_optimization_bug,
                    "min_failing_opt_level": attr_result.min_failing_opt_level,
                    "fired_passes": attr_result.fired_passes,
                    "opt_level_results": {
                        str(k): v.symptom.value if v else "pass"
                        for k, v in attr_result.opt_level_results.items()
                    },
                    "summary": attr_result.summary,
                }
                with open(bug_dir / "attribution.json", "w") as f:
                    json.dump(attribution_data, f, indent=2)
            except Exception as e:
                logger.warning("Attribution failed: %s", e)

        report = {
            "symptom": bug.symptom.value,
            "severity": bug.severity.value,
            "max_abs_diff": bug.max_abs_diff,
            "details": bug.details,
            "fired_passes": bug.fired_passes,
            "error_log": bug.error_log,
            "n_mutations_full": len(records),
            "n_mutations_reduced": len(reduced_records),
        }
        if attribution_data:
            report["is_optimization_bug"] = attribution_data.get("is_optimization_bug", False)
        with open(bug_dir / "bug_report.json", "w") as f:
            json.dump(report, f, indent=2)

        np.savez(bug_dir / "inputs.npz", **inputs)
        logger.info("Bug saved to %s", bug_dir)

    # ── Legacy path (GraphIR) ───────────────────────────────

    def _fuzz_one_legacy(self, iteration: int):
        """Execute one fuzzing iteration."""
        # 1. Generate seed
        ir, model = self.seed_gen.generate()
        self.stats["total_seeds"] += 1

        # Materialize original (seed) model for metamorphic comparison
        original_onnx = model.native_model
        original_inputs = make_random_input(model.input_like)

        # 2. Mutate
        n_mutations = randint(1, self.max_mutations)
        mutated_ir, records = self.mutator.mutate(ir, n_mutations=n_mutations)
        self.stats["total_mutations"] += len(records)

        # 3. Materialize mutated model
        mutated_model = ONNXModelCPU.from_gir(mutated_ir)
        mutated_model.refine_weights()
        mutated_onnx = mutated_model.native_model

        mutated_shape_dict = {
            name: list(aten.shape) for name, aten in mutated_model.input_like.items()
        }
        mutated_inputs = make_random_input(mutated_model.input_like)

        # 4a. Metamorphic test: verify G ≈ G' on ORT, then G' on ORT vs Relax
        # This is the primary test — it catches compiler bugs while filtering
        # out RPC equivalence violations (framework issues).
        bug = self.oracle.compare_metamorphic(
            original_onnx, mutated_onnx,
            original_inputs, mutated_inputs,
            mutated_shape_dict=mutated_shape_dict,
        )

        # 4b. Fallback cross-backend test if metamorphic test passed
        # This catches bugs that don't require the original model for detection
        # (e.g., compilation crashes on G' alone).
        if bug is None:
            bug = self.oracle.compare(
                mutated_onnx, mutated_inputs,
                shape_dict=mutated_shape_dict,
            )

        # 5. If bug found, reduce and save
        if bug is not None:
            self.stats["bugs_found"] += 1
            logger.info(
                "Bug #%d found at iteration %d: %s (severity=%s)",
                self.stats["bugs_found"], iteration, bug.symptom.value, bug.severity.value,
            )

            # 6. Attempt delta debugging reduction
            reduced_records = records
            if self.enable_reduction and len(records) > 1:
                reduced_records = self._reduce(ir, records)

            self._save_bug(
                bug, ir, mutated_ir, records, reduced_records,
                mutated_inputs, mutated_shape_dict, mutated_onnx, iteration,
            )

    def _reduce(
        self,
        seed_ir: GraphIR,
        records: List[MutationRecord],
    ) -> List[MutationRecord]:
        """Minimize the mutation set that reproduces the bug via delta debugging."""

        def reproduces_bug(candidate_ir: GraphIR) -> Optional[BugReport]:
            """Predicate: does this GraphIR still trigger a bug?"""
            try:
                # Materialize original seed model
                seed_model = ONNXModelCPU.from_gir(seed_ir)
                seed_model.refine_weights()
                seed_onnx = seed_model.native_model
                seed_inputs = make_random_input(seed_model.input_like)

                # Materialize candidate (partially-mutated) model
                model = ONNXModelCPU.from_gir(candidate_ir)
                model.refine_weights()
                onnx_proto = model.native_model
                shape_dict = {
                    name: list(aten.shape) for name, aten in model.input_like.items()
                }
                inputs = make_random_input(model.input_like)

                # Use metamorphic comparison: ensures RPC equivalence
                # still holds for the reduced mutation subset
                return self.oracle.compare_metamorphic(
                    seed_onnx, onnx_proto,
                    seed_inputs, inputs,
                    mutated_shape_dict=shape_dict,
                )
            except Exception as e:
                logger.debug("Reduction predicate error: %s", e)
                return None

        debugger = DeltaDebugger(seed_ir, records, reproduces_bug)
        try:
            minimized = debugger.minimize()
            if len(minimized) < len(records):
                self.stats["bugs_reduced"] += 1
                logger.info(
                    "Reduced %d -> %d mutation records",
                    len(records), len(minimized),
                )
            return minimized
        except Exception as e:
            logger.warning("Delta debugging failed: %s", e)
            return records

    def _save_bug(
        self,
        bug: BugReport,
        seed_ir,
        mutated_ir,
        records: List[MutationRecord],
        reduced_records: List[MutationRecord],
        inputs: Dict[str, np.ndarray],
        shape_dict: Optional[Dict[str, List[int]]],
        onnx_proto,
        iteration: int,
    ):
        """Save bug report to disk, optionally with attribution."""
        bug_dir = self.output_dir / f"bug_{self.stats['bugs_found']:04d}_iter{iteration}"
        bug_dir.mkdir(parents=True, exist_ok=True)

        # Save GraphIR DOT visualizations
        with open(bug_dir / "seed_ir.dot", "w") as f:
            f.write(seed_ir.to_dot())
        with open(bug_dir / "mutated_ir.dot", "w") as f:
            f.write(mutated_ir.to_dot())

        # Save full mutation records
        records_data = [asdict(r) for r in records]
        with open(bug_dir / "mutation_records.json", "w") as f:
            json.dump(records_data, f, indent=2)

        # Save reduced records (if different from full)
        if len(reduced_records) < len(records):
            reduced_data = [asdict(r) for r in reduced_records]
            with open(bug_dir / "mutation_records_reduced.json", "w") as f:
                json.dump(reduced_data, f, indent=2)

        # Optimization attribution via opt-level ablation
        attribution_data = {}
        if self.enable_attribution:
            try:
                attr_result = attribute_bug(onnx_proto, inputs, shape_dict)
                attribution_data = {
                    "is_optimization_bug": attr_result.is_optimization_bug,
                    "min_failing_opt_level": attr_result.min_failing_opt_level,
                    "fired_passes": attr_result.fired_passes,
                    "opt_level_results": {
                        str(k): v.symptom.value if v else "pass"
                        for k, v in attr_result.opt_level_results.items()
                    },
                    "summary": attr_result.summary,
                }
                with open(bug_dir / "attribution.json", "w") as f:
                    json.dump(attribution_data, f, indent=2)
                logger.info("Attribution: %s", attr_result.summary)
            except Exception as e:
                logger.warning("Attribution failed: %s", e)

        # Save bug report
        report = {
            "symptom": bug.symptom.value,
            "severity": bug.severity.value,
            "max_abs_diff": bug.max_abs_diff,
            "details": bug.details,
            "fired_passes": bug.fired_passes,
            "error_log": bug.error_log,
            "n_mutations_full": len(records),
            "n_mutations_reduced": len(reduced_records),
        }
        if attribution_data:
            report["is_optimization_bug"] = attribution_data.get("is_optimization_bug", False)
        with open(bug_dir / "bug_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Save inputs
        np.savez(bug_dir / "inputs.npz", **inputs)

        logger.info("Bug saved to %s", bug_dir)

    def _save_stats(self):
        """Save fuzzing statistics."""
        stats_path = self.output_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)
