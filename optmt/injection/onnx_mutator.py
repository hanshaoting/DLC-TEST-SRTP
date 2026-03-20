from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import onnx
from onnx import checker, helper, shape_inference

from optmt.injection.input_bridge import plan_bridges
from optmt.injection.onnx_rpc import RPC_REGISTRY, stitch
from optmt.injection.onnx_utils import (
    collect_prefix_context,
    get_tensor_info,
    prefix_graph,
    replace_input_references,
    topological_sort,
)
from optmt.injection.pattern_pool import PatternEntry, PatternPool

logger = logging.getLogger(__name__)

# Only select float-type tensors as anchors (RPC requires float semantics).
_FLOAT_DTYPES = {
    onnx.TensorProto.FLOAT,
    onnx.TensorProto.DOUBLE,
    onnx.TensorProto.FLOAT16,
}


@dataclass
class OnnxMutationRecord:
    """Record of a single ONNX-native injection."""

    anchor_tensor: str
    pattern_name: str
    pattern_rel_path: str
    rpc_strategy: str
    bridge_strategies: List[str]
    mutation_index: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "OnnxMutationRecord":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> "OnnxMutationRecord":
        return cls.from_dict(json.loads(s))


class OnnxMutator:
    """Orchestrates ONNX-native pattern injection with RPC equivalence."""

    def __init__(
        self,
        pool: PatternPool,
        strategies: Optional[List[str]] = None,
        rng: Optional[random.Random] = None,
    ):
        self._pool = pool
        self._strategies = strategies or list(RPC_REGISTRY.keys())
        self._rng = rng or random.Random()

    def mutate(
        self,
        model: onnx.ModelProto,
        n_mutations: int = 1,
    ) -> Tuple[onnx.ModelProto, List[OnnxMutationRecord]]:
        """Apply `n_mutations` sequential injections with random choices.

        Returns:
            (mutated_model, records)

        任务 5：请完成该函数。

        目标：
        对同一个 ONNX 模型连续执行若干次注入，每次注入都随机选择：
        - anchor tensor
        - pattern
        - RPC strategy

        推荐实现步骤：
        1. 定义一个小的重试上限，例如：
           - `MAX_INJECT_RETRIES = 3`
        2. 初始化 `records = []`
        3. 对 `range(n_mutations)` 逐轮执行：
           a. 先对当前 `model` 做 shape inference
           b. 调用 `_select_anchor(model.graph)` 选择一个 anchor
           c. 进入重试循环：
              - 从 pattern pool 随机采样 1 个 pattern
              - 从 `self._strategies` 中随机选 1 个 RPC 策略
              - 调用 `_inject(...)`
              - 若成功：
                * 更新 `model`
                * 收集 `record`
                * `break`
              - 若失败：
                * 如果还没到最后一次重试，则记录 debug 日志后继续
                * 如果已经到最后一次，则把异常重新抛出
        4. 返回 `(model, records)`

        设计意图：
        - 某些 pattern + anchor + strategy 组合可能会失败
        - 增加少量重试可显著降低整个 fuzzing 流程的 error rate

        测试关注点：
        - 单次 mutation 后模型能通过 checker
        - 多次 stacked mutation 后仍能通过 checker
        - `records` 数量与 `n_mutations` 一致
        - 每条 record 的 `mutation_index` 正确
        """
        raise NotImplementedError("TODO: implement OnnxMutator.mutate")

    def replay(
        self,
        model: onnx.ModelProto,
        records: List[OnnxMutationRecord],
    ) -> onnx.ModelProto:
        """Replay a specific list of mutation records onto the model.

        This is mainly used by reducer / debugger logic to deterministically
        re-apply a known sequence of mutations.

        任务 5：请完成该函数。

        目标：
        按照给定的 mutation records，确定性地重新注入同一组变异。

        推荐实现步骤：
        1. 遍历 `enumerate(records)`
        2. 对每条 `record`：
           a. 调用 `_find_pattern(record.pattern_rel_path)`
           b. 如果返回 `None`，抛出 `ValueError`
           c. 调用 `_inject(...)`：
              - anchor 使用 `record.anchor_tensor`
              - pattern 使用找到的 `entry`
              - strategy 使用 `record.rpc_strategy`
              - mutation_idx 使用当前循环下标 `i`
           d. 更新 `model`
        3. 返回最终模型

        注意：
        - replay 强调“确定性”，所以不能重新随机选 pattern 或 strategy
        - 这里通常不必信任 `record.pattern_name`，以 `pattern_rel_path` 为准更稳妥
        """
        raise NotImplementedError("TODO: implement OnnxMutator.replay")

    # ── core injection logic (shared by mutate & replay) ─────

    def _inject(
        self,
        model: onnx.ModelProto,
        anchor: str,
        entry: PatternEntry,
        strategy: str,
        mutation_idx: int,
    ) -> Tuple[onnx.ModelProto, OnnxMutationRecord]:
        """Inject one prefixed pattern subgraph around a chosen anchor.

        任务 5：请完成该函数。

        这是整个 Stage 3 的核心编排函数，建议严格按照下面步骤实现。

        目标：
        将一个 pattern 子图通过“输入桥接 + RPC 等价拼接”的方式注入到
        `model` 的某个 anchor tensor 处，并返回：
        - 新模型 `new_model`
        - 对应的 `OnnxMutationRecord`

        推荐实现步骤：

        Step 0. 预处理
        - 先对 `model` 做 shape inference
        - 取出 `graph = model.graph`

        Step 1. 查询 anchor 信息
        - 调用 `get_tensor_info(graph, anchor)`
        - 若结果为 `None`，抛出 `ValueError`
        - 拿到：
          - `a_shape`
          - `a_dtype`

        Step 2. 命名空间隔离 pattern
        - 构造前缀：`prefix = f"p{mutation_idx}_"`
        - 调用 `prefix_graph(entry.graph, prefix)` 得到 `pat`

        Step 3. 尝试对隔离后的 pattern 做 shape inference
        - 用 `helper.make_model(...)` 把 `pat` 包装成临时模型
        - opset 可使用 `entry.opset`
        - 调用 `shape_inference.infer_shapes(..., check_type=True)`
        - 成功则用 `pat_model.graph` 覆盖 `pat`
        - 若失败，可选择忽略异常并继续（保持鲁棒性）

        Step 4. 规划 pattern 输入桥接
        - 调用 `collect_prefix_context(graph, anchor)` 得到 `ctx`
        - 调用 `plan_bridges(graph, pat, ctx, prefix, self._rng)` 得到 `plans`

        Step 5. 确定 pattern 输出信息
        - 取 pattern 第一个输出：`pat.output[0].name`
        - 调用 `get_tensor_info(pat, pat_out_name)`
        - 若拿到 shape，则把其中非正维替换成一个具体默认值（例如 2）
        - 若拿不到，退化为 `[1]`
        - 得到 `p_shape`

        Step 6. 生成 RPC 拼接结构
        - 调用 `stitch(...)`
        - 传入：
          - `strategy`
          - `anchor`
          - `a_shape`
          - `a_dtype`
          - `pat_out_name`
          - `p_shape`
          - `prefix`
        - 得到 `rpc`

        Step 7. 改写原图中 anchor 的下游引用
        - 在真正 graft 新节点之前，先调用：
          `replace_input_references(graph, anchor, rpc.new_anchor_name)`
        - 这一步会让原来消费 anchor 的地方改为消费 RPC 新输出

        Step 8. 改写 pattern 自由输入 -> bridge 输出
        - 遍历每个 `plan`
        - 若 `plan.bridge_output != plan.pattern_input_name`
        - 就遍历 `pat.node` 的输入，把等于 `plan.pattern_input_name` 的地方
          替换成 `plan.bridge_output`

        Step 9. Graft 到原图
        - 先把所有 bridge 的 nodes / initializers 加入原图
        - 再把 `pat.node` / `pat.initializer` 加入原图
        - 最后把 `rpc.nodes` / `rpc.initializers` 加入原图

        Step 10. 验证新图
        - 调用 `topological_sort(graph)`
        - 用 `helper.make_model(...)` 重新包成模型
        - 记得继承原模型的 `opset_import`
        - 最好保留 `ir_version`
        - 再调用 `shape_inference.infer_shapes(..., check_type=True)`
        - 再调用 `checker.check_model(...)`

        Step 11. 生成 mutation record
        - 字段建议如下：
          - `anchor_tensor=anchor`
          - `pattern_name=entry.name`
          - `pattern_rel_path=entry.rel_path`
          - `rpc_strategy=strategy`
          - `bridge_strategies=[p.strategy for p in plans]`
          - `mutation_index=mutation_idx`

        Step 12. 返回
        - `return new_model, record`

        实现注意事项：
        - graph graft 的顺序不要乱，通常 bridge -> pattern -> rpc 更直观
        - `replace_input_references()` 要在 graft 新节点前做，避免把 RPC 自己的输入也误改
        - `checker.check_model()` 是最后的强约束，任何非法 ONNX 都应在此暴露
        - `pat.output[0]` 的假设建立在 pattern library 至少有一个输出之上
        """
        raise NotImplementedError("TODO: implement OnnxMutator._inject")

    # ── helpers ──────────────────────────────────────────────

    def _select_anchor(self, graph: onnx.GraphProto) -> str:
        """Pick a random intermediate tensor as injection anchor.

        只考虑满足以下条件的 tensor：
        - 来自某个 node 的输出
        - 不是 graph input
        - 不是 initializer
        - shape 已知且各维度 > 0
        - dtype 属于浮点类型集合 `_FLOAT_DTYPES`

        额外策略：
        - 优先选择“不是 graph output”的中间张量
        - 如果所有候选都是 graph output，则退化为从全部候选中选
        - 若不存在任何合法候选，抛出 `ValueError`

        任务 5：请完成该函数。

        推荐实现步骤：
        1. 先收集：
           - `graph_inputs`
           - `init_names`
           - `output_names`
        2. 遍历 `graph.node` 的每个 output
        3. 过滤掉空名字、graph input、initializer
        4. 调用 `get_tensor_info(graph, out)` 获取 `(shape, dtype)`
        5. 只保留：
           - `shape` 非空
           - 所有维度 > 0
           - `dtype in _FLOAT_DTYPES`
        6. 构造 `candidates`
        7. 再从中筛出 `non_output = [c for c in candidates if c not in output_names]`
        8. 若 `non_output` 非空，优先从它随机选
           否则从 `candidates` 中随机选
        9. 若最终没有候选，抛出 `ValueError("No valid anchor candidates in graph")`

        说明：
        - 之所以优先选非 graph output，是为了尽量减少对模型最终接口的扰动
        - 但当图很浅时，可能只能选 graph output
        """
        raise NotImplementedError("TODO: implement OnnxMutator._select_anchor")

    def _find_pattern(self, rel_path: str) -> Optional[PatternEntry]:
        """Find a pattern entry in the pool by relative path.

        任务 5：请完成该函数。

        目标：
        在 `self._pool.entries` 中找到 `entry.rel_path == rel_path` 的 pattern。

        推荐实现步骤：
        1. 遍历 `self._pool.entries`
        2. 如果某个 `entry.rel_path == rel_path`，返回该 `entry`
        3. 否则返回 `None`

        说明：
        - replay 阶段会用这个函数恢复 record 中引用的 pattern
        - 这是一个很短但很实用的小工具函数
        """
        raise NotImplementedError("TODO: implement OnnxMutator._find_pattern")
