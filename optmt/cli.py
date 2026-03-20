"""CLI entry point for the OptMT fuzzing framework."""

from __future__ import annotations

import argparse
import logging
import sys
from random import Random

from optmt.backend.tvm_relax import TVMRelax
from optmt.mutator import Mutator
from optmt.oracle.differential import DifferentialOracle
from optmt.runner import FuzzRunner
from optmt.seed.generator import SeedGenerator, SeedConfig


def main():
    parser = argparse.ArgumentParser(
        description="OptMT: Optimization-Aware Metamorphic Testing for TVM Relax"
    )
    parser.add_argument(
        "--time-budget", type=int, default=300,
        help="Fuzzing time budget in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=None,
        help="Maximum number of fuzzing iterations",
    )
    parser.add_argument(
        "--max-nodes", type=int, default=10,
        help="Maximum nodes in seed graph (default: 10)",
    )
    parser.add_argument(
        "--max-mutations", type=int, default=3,
        help="Maximum mutations per seed (default: 3)",
    )
    parser.add_argument(
        "--opt-level", type=int, default=3,
        help="TVM optimization level (default: 3)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="fuzz_output",
        help="Output directory for bug reports (default: fuzz_output)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--onnx-native", action="store_true",
        help="Enable ONNX-native injection path (requires pattern library)",
    )
    parser.add_argument(
        "--pattern-library", type=str, default="pattern_library",
        help="Path to the pattern library directory (default: pattern_library)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build components
    seed_gen = SeedGenerator(
        config=SeedConfig(max_nodes=args.max_nodes, seed=args.seed)
    )
    mutator = Mutator(seed=args.seed)
    backend = TVMRelax(target="llvm", opt_level=args.opt_level)
    oracle = DifferentialOracle(backend=backend)

    onnx_mutator = None
    if args.onnx_native:
        from optmt.injection.pattern_pool import PatternPool
        from optmt.injection.onnx_mutator import OnnxMutator

        pool = PatternPool(root=args.pattern_library)
        pool.load_all()
        onnx_mutator = OnnxMutator(pool, rng=Random(args.seed))

    runner = FuzzRunner(
        seed_gen=seed_gen,
        mutator=mutator,
        oracle=oracle,
        output_dir=args.output_dir,
        max_mutations=args.max_mutations,
        onnx_mutator=onnx_mutator,
    )

    runner.run(
        time_budget_s=args.time_budget,
        max_iterations=args.max_iterations,
    )


if __name__ == "__main__":
    main()
