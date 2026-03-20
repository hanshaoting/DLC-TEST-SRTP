"""Pass Probe: instrument TVM Relax compilation to detect which passes fire."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import tvm
import tvm.ir
import tvm.ir.instrument


@dataclass
class PassRecord:
    """Record of a single pass execution."""

    pass_name: str
    changed: bool


@tvm.ir.instrument.pass_instrument
class PassProbe:
    """TVM PassInstrument that records which passes modify the IR.

    Usage:
        probe = PassProbe()
        with tvm.transform.PassContext(opt_level=3, instruments=[probe]):
            ex = relax.build(mod, target="llvm")
        fired = probe.get_fired_passes()
    """

    def __init__(self):
        self.records: List[PassRecord] = []
        self._before_mod = None

    def run_before_pass(self, mod, info):
        self._before_mod = mod

    def run_after_pass(self, mod, info):
        changed = not tvm.ir.structural_equal(self._before_mod, mod)
        self.records.append(PassRecord(pass_name=info.name, changed=changed))
        self._before_mod = None

    def get_fired_passes(self) -> List[str]:
        """Return names of passes that actually modified the IR."""
        return [r.pass_name for r in self.records if r.changed]

    def get_all_passes(self) -> List[str]:
        """Return names of all passes that were executed."""
        return [r.pass_name for r in self.records]

    def reset(self):
        """Clear all recorded data."""
        self.records = []
        self._before_mod = None
