"""Delta Debugging: minimize mutation records that trigger a bug.

Implements the ddmin algorithm to find the minimal subset of mutation records
that still reproduces a detected bug.

Two debugger classes:
  - DeltaDebugger: legacy GraphIR-level replay
  - OnnxDeltaDebugger: ONNX-native replay via OnnxMutator.replay()
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Callable, List, Optional

from typing import TYPE_CHECKING

from nnsmith.gir import GraphIR

from optmt.mutator import MutationRecord

if TYPE_CHECKING:
    from optmt.oracle.differential import BugReport

logger = logging.getLogger(__name__)


class DeltaDebugger:
    """Minimize the set of mutations that reproduce a compiler bug.

    Uses the classic ddmin algorithm: binary search over MutationRecords,
    replaying subsets onto the seed_ir to find the minimal triggering set.
    """

    def __init__(
        self,
        seed_ir: GraphIR,
        records: List[MutationRecord],
        reproduces_bug: Callable[[GraphIR], Optional[BugReport]],
    ):
        """
        Args:
            seed_ir: The original (unmutated) GraphIR.
            records: Full list of MutationRecords that triggered the bug.
            reproduces_bug: A callable that takes a mutated GraphIR and returns
                            a BugReport if the bug reproduces, else None.
        """
        self.seed_ir = seed_ir
        self.records = records
        self.reproduces_bug = reproduces_bug

    def minimize(self) -> List[MutationRecord]:
        """Run ddmin to find the minimal error-inducing record subset.

        Returns:
            Minimal list of MutationRecords that still trigger the bug.
        """
        if len(self.records) <= 1:
            return self.records

        indices = list(range(len(self.records)))
        minimal_indices = self._ddmin(indices, [])
        return [self.records[i] for i in minimal_indices]

    def _ddmin(self, delta_ids: List[int], remaining: List[int]) -> List[int]:
        """Recursive ddmin algorithm.

        Args:
            delta_ids: Current candidate set of record indices.
            remaining: Indices that must be kept (complement context).

        Returns:
            Minimal subset of delta_ids that reproduces the bug.
        """
        if len(delta_ids) <= 1:
            return delta_ids

        half = len(delta_ids) // 2
        left = delta_ids[:half]
        right = delta_ids[half:]

        # Try left half + remaining
        left_with_ctx = sorted(set(left + remaining))
        if self._test_subset(left_with_ctx):
            return self._ddmin(left, remaining)

        # Try right half + remaining
        right_with_ctx = sorted(set(right + remaining))
        if self._test_subset(right_with_ctx):
            return self._ddmin(right, remaining)

        # Bug requires elements from both halves — recurse on each
        left_inducing = self._ddmin(left, sorted(set(right + remaining)))
        right_inducing = self._ddmin(
            right, sorted(set(left_inducing + remaining))
        )
        return sorted(set(left_inducing + right_inducing))

    def _test_subset(self, record_indices: List[int]) -> bool:
        """Replay a subset of mutations and check if the bug reproduces.

        Args:
            record_indices: Indices into self.records to replay.

        Returns:
            True if the bug is reproduced with this subset.
        """
        mutated_ir = deepcopy(self.seed_ir)

        for idx in sorted(record_indices):
            rec = self.records[idx]
            try:
                # Re-apply the mutation
                # Note: we need the pattern and strategy instances
                from optmt.pattern import ALL_PATTERNS
                from optmt.rpc import ALL_RPC_STRATEGIES

                pattern = None
                for p_cls in ALL_PATTERNS:
                    p = p_cls() if not hasattr(p_cls, 'info') else p_cls()
                    if p.info.name == rec.pattern_name:
                        pattern = p
                        break

                strategy = None
                for s_cls in ALL_RPC_STRATEGIES:
                    s = s_cls() if callable(s_cls) else s_cls
                    if s.name == rec.rpc_name:
                        strategy = s
                        break

                if pattern is None or strategy is None:
                    logger.warning("Cannot find pattern=%s or rpc=%s", rec.pattern_name, rec.rpc_name)
                    return False

                # Check if target_var still exists
                if rec.target_var not in mutated_ir.vars:
                    return False

                strategy.apply(mutated_ir, rec.target_var, pattern)
                mutated_ir.wellform_repair()
            except Exception as e:
                logger.debug("Replay failed for record %d: %s", idx, e)
                return False

        # Test if bug reproduces
        bug = self.reproduces_bug(mutated_ir)
        return bug is not None


class OnnxDeltaDebugger:
    """Minimize ONNX-native mutation records via ddmin.

    Replays record subsets using OnnxMutator.replay() and tests
    with a predicate that receives the replayed ONNX ModelProto.
    """

    def __init__(
        self,
        seed_model,
        records: List,
        mutator,
        reproduces_bug: Callable,
    ):
        """
        Args:
            seed_model: The original (unmutated) ONNX ModelProto.
            records: Full list of OnnxMutationRecords.
            mutator: OnnxMutator with replay() capability.
            reproduces_bug: Callable(ModelProto) -> BugReport or None.
        """
        self.seed_model = seed_model
        self.records = records
        self.mutator = mutator
        self.reproduces_bug = reproduces_bug

    def minimize(self) -> List:
        """Run ddmin. Returns minimal record subset."""
        if len(self.records) <= 1:
            return self.records

        indices = list(range(len(self.records)))
        minimal = self._ddmin(indices, [])
        return [self.records[i] for i in minimal]

    def _ddmin(self, delta_ids: List[int], remaining: List[int]) -> List[int]:
        if len(delta_ids) <= 1:
            return delta_ids

        half = len(delta_ids) // 2
        left = delta_ids[:half]
        right = delta_ids[half:]

        left_with_ctx = sorted(set(left + remaining))
        if self._test_subset(left_with_ctx):
            return self._ddmin(left, remaining)

        right_with_ctx = sorted(set(right + remaining))
        if self._test_subset(right_with_ctx):
            return self._ddmin(right, remaining)

        left_inducing = self._ddmin(left, sorted(set(right + remaining)))
        right_inducing = self._ddmin(
            right, sorted(set(left_inducing + remaining))
        )
        return sorted(set(left_inducing + right_inducing))

    def _test_subset(self, record_indices: List[int]) -> bool:
        """Replay a record subset and test for bug reproduction."""
        subset = [self.records[i] for i in sorted(record_indices)]
        try:
            mutated = self.mutator.replay(deepcopy(self.seed_model), subset)
        except Exception as e:
            logger.debug("ONNX replay failed: %s", e)
            return False

        bug = self.reproduces_bug(mutated)
        return bug is not None
