from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from optmt.injection.onnx_utils import get_tensor_info

# Default concrete value for symbolic (unknown) dimensions.
_SYMBOLIC_DIM_DEFAULT = 2


@dataclass
class BridgePlan:
    """Describes how one pattern input is connected to the seed graph."""

    pattern_input_name: str
    source_tensor: str
    bridge_nodes: List[onnx.NodeProto] = field(default_factory=list)
    bridge_initializers: List[onnx.TensorProto] = field(default_factory=list)
    bridge_output: str = ""
    strategy: str = ""  # "reuse" or "bridge"


def plan_bridges(
    seed_graph: onnx.GraphProto,
    pattern_graph: onnx.GraphProto,
    prefix_context: Set[str],
    prefix: str,
    rng: Optional[random.Random] = None,
) -> List[BridgePlan]:
    """Plan input bridges for every free input of a prefixed pattern graph.

    Free inputs = pattern graph.input minus initializer names.

    Parameters
    ----------
    seed_graph : the original seed graph, used for tensor info lookup
    pattern_graph : the namespace-isolated pattern subgraph
    prefix_context : set of tensor names available at the anchor point
    prefix : the namespace prefix (e.g. "p0_") for unique bridge names
    rng : random source

    Returns
    -------
    list of BridgePlan, one per free input

    任务 3：请完成该函数。

    目标：
    为 pattern 的每个“自由输入”制定接入 seed graph 的方案。

    需要完成的核心逻辑：
    1. 找出 free inputs：
       - `pattern_graph.input` 中去掉那些同时也是 initializer 的输入名
    2. 调用 `_collect_available_tensors()` 收集 prefix context 里可用的 tensor
    3. 对每个 free input：
       - 调用 `_resolve_target()` 提取目标 shape/dtype
       - 先尝试 `_find_exact_match()`：
         - 如果找到 shape 和 dtype 都完全相同的 tensor
         - 则生成一个 `BridgePlan(strategy="reuse")`
       - 否则进入桥接策略：
         - 用 `_pick_source()` 从 available 中随机挑一个 source
         - 若没有可用 source，抛出 `ValueError`
         - 调用 `_build_bridge_chain()` 生成桥接节点链
         - 返回 `BridgePlan(strategy="bridge")`
    4. 最终返回所有输入对应的 plan 列表

    实现提示：
    - 这里的 `pattern_graph` 已经是 prefix 后的图，所以输入名也带前缀
    - `bridge_output` 对于 reuse 情况应直接等于 `source_tensor`
    - `bridge_output` 对于 bridge 情况应等于桥接链最后一个节点的输出名
    - 测试会检查：
      - 完全匹配时是否走 reuse
      - shape/dtype 不匹配时是否构建 bridge
      - 生成的桥接链是否可做 shape inference / ORT 执行
    """
    raise NotImplementedError("TODO: implement plan_bridges")


# ── internal helpers ──────────────────────────────────────────────


def _collect_available_tensors(
    graph: onnx.GraphProto, prefix_context: Set[str]
) -> Dict[str, Tuple[List[int], int]]:
    """Gather {name: (shape, dtype)} for context tensors with known info.

    任务 3：请完成该函数。

    目标：
    从 prefix context 中筛出“已知 shape 且可用”的 tensor。

    推荐实现步骤：
    1. 初始化空字典 `result`
    2. 遍历 `prefix_context` 中每个名字
    3. 调用 `get_tensor_info(graph, name)`：
       - 若返回 `None`，跳过
       - 否则得到 `(shape, dtype)`
    4. 只保留满足以下条件的 tensor：
       - `shape` 非空
       - 所有维度都大于 0
    5. 写入 `result[name] = (shape, dtype)`
    6. 返回结果字典

    说明：
    - 这里故意过滤未知维度/非法维度，避免后续桥接时无法确定元素个数
    - 返回值会被后面的 exact match 与随机选源逻辑复用
    """
    raise NotImplementedError("TODO: implement _collect_available_tensors")


def _resolve_target(
    vi: onnx.ValueInfoProto,
) -> Tuple[List[int], int]:
    """Extract concrete (shape, dtype) from a pattern input ValueInfoProto.

    任务 3：请完成该函数。

    目标：
    将 pattern 输入的 `ValueInfoProto` 解析成可直接用于桥接的目标 shape/dtype。

    推荐实现步骤：
    1. 取出 `vi.type.tensor_type`
    2. 读取 dtype：
       - 若 `elem_type != 0`，直接使用它
       - 否则默认用 `TensorProto.FLOAT`
    3. 读取 shape：
       - 若存在 `shape` 字段，则逐维处理
       - 若某一维 `dim_value > 0`，使用真实值
       - 否则使用 `_SYMBOLIC_DIM_DEFAULT`
    4. 如果整个 tensor 连 shape 字段都没有：
       - 可退化为 `[_SYMBOLIC_DIM_DEFAULT]`
    5. 返回 `(shape, dtype)`

    设计原因：
    - pattern 里可能含有 symbolic/unknown 维
    - 为了让后续桥接链能真正构造出 ONNX 常量，需要把它 concretize
    """
    raise NotImplementedError("TODO: implement _resolve_target")


def _find_exact_match(
    available: Dict[str, Tuple[List[int], int]],
    target_shape: List[int],
    target_dtype: int,
) -> Optional[str]:
    """Find a tensor whose shape and dtype exactly match the target.

    任务 3：请完成该函数。

    目标：
    在可用 tensor 集合中找一个“shape 完全相同且 dtype 完全相同”的 tensor。

    推荐实现步骤：
    1. 遍历 `available.items()`
    2. 对每个 `(name, (shape, dtype))`：
       - 若 `shape == target_shape` 且 `dtype == target_dtype`
       - 立即返回该 `name`
    3. 如果遍历结束仍未命中，返回 `None`

    说明：
    - 这是最低成本的接线方式，对应 `reuse`
    - 不需要做近似匹配，也不需要考虑 broadcast
    """
    raise NotImplementedError("TODO: implement _find_exact_match")


def _pick_source(
    available: Dict[str, Tuple[List[int], int]],
    rng: random.Random,
) -> Optional[str]:
    """Randomly choose one source tensor from available tensors.

    任务 3：请完成该函数。

    目标：
    当无法 exact match 时，从 available 里随机挑一个 source tensor 做桥接。

    推荐实现步骤：
    1. 若 `available` 为空，返回 `None`
    2. 否则对 `list(available.keys())` 调用 `rng.choice(...)`
    3. 返回选中的 tensor 名

    说明：
    - 这里不做启发式最优选择，保持实现简单
    - 随机性由外部传入的 `rng` 控制，便于测试复现
    """
    raise NotImplementedError("TODO: implement _pick_source")


def _build_bridge_chain(
    *,
    source_name: str,
    source_shape: List[int],
    source_dtype: int,
    target_shape: List[int],
    target_dtype: int,
    pattern_input_name: str,
    prefix: str,
    bridge_idx: int,
) -> BridgePlan:
    """Build Cast -> Reshape(1D) -> Slice/Pad -> Reshape(target) chain.

    任务 3：请完成该函数。

    目标：
    从一个 seed tensor 构造出能喂给 pattern 输入的桥接张量。

    总体策略：
    - 如果 dtype 不同，先插入 `Cast`
    - 如果 shape 已相同，可直接结束
    - 如果元素总数相同，用一次 `Reshape`
    - 如果元素总数不同：
      1. 先 flatten 到 1D
      2. 若源元素更多，使用 `Slice` 截断
      3. 若源元素更少，使用 `Pad` 补零
      4. 最后 `Reshape` 到目标 shape

    推荐实现步骤：
    1. 初始化：
       - `nodes = []`
       - `inits = []`
       - `cur = source_name`
       - `cur_dtype = source_dtype`
       - `cur_shape = list(source_shape)`
       - `tag = f"{prefix}br{bridge_idx}_"`

    2. 处理 dtype：
       - 若 `cur_dtype != target_dtype`
       - 新建 `Cast` 节点
       - 更新 `cur` 为 Cast 输出名
       - 更新 `cur_dtype = target_dtype`

    3. 计算元素总数：
       - `src_numel = math.prod(cur_shape)`
       - `tgt_numel = math.prod(target_shape)`

    4. 分情况处理 shape：
       - 若 `cur_shape == target_shape`：
         - 不需要新增 shape 相关节点
       - 若 `src_numel == tgt_numel`：
         - 调用 `_add_reshape(cur, target_shape, tag, "eq")`
       - 否则：
         a. 先 flatten 到 `[src_numel]`
         b. 若 `src_numel > tgt_numel`：
            - 构造 `Slice`
            - 常量包括 `starts=[0]`, `ends=[tgt_numel]`, `axes=[0]`
         c. 若 `src_numel < tgt_numel`：
            - 构造 `Pad`
            - 对 1D 张量可用 `pads=[0, tgt_numel - src_numel]`
            - `mode="constant"`
         d. 最后 reshape 到 `target_shape`

    5. 返回 `BridgePlan`：
       - `pattern_input_name=pattern_input_name`
       - `source_tensor=source_name`
       - `bridge_nodes=nodes`
       - `bridge_initializers=inits`
       - `bridge_output=cur`
       - `strategy="bridge"`（若实际没加节点也可视为 reuse，但本题建议按原逻辑处理）

    命名提示：
    - 输出 tensor 名、常量名、node 名都应带 `tag`
    - 例如：`f"{tag}cast"`, `f"{tag}pads"`, `f"{tag}n_pad"`

    易错点：
    - `Pad` / `Slice` 所需常量都是 int64
    - `Cast` 常量不需要 initializer；`Cast` 的目标类型通过属性 `to=...`
    - 最终 `bridge_output` 一定要是“最后可用 tensor 名”
    """
    raise NotImplementedError("TODO: implement _build_bridge_chain")


def _add_reshape(
    input_name: str,
    shape: List[int],
    tag: str,
    label: str,
) -> Tuple[str, List[onnx.NodeProto], List[onnx.TensorProto]]:
    """Create a Reshape node + shape constant. Returns (out_name, nodes, inits).

    任务 3：请完成该函数。

    目标：
    封装一个小工具函数，减少重复的 Reshape 构造代码。

    推荐实现步骤：
    1. 构造输出名：
       - `out = f"{tag}rs_{label}"`
    2. 构造 shape 常量名：
       - `shape_c = f"{tag}rs_{label}_shape"`
    3. 用 `numpy_helper.from_array(np.array(shape, dtype=np.int64), name=shape_c)`
       创建 initializer
    4. 用 `helper.make_node("Reshape", [input_name, shape_c], [out], ...)`
       创建节点
    5. 返回：
       - 输出名 `out`
       - 单元素节点列表 `[node]`
       - 单元素 initializer 列表 `[init]`

    说明：
    - 这个函数虽然短，但在 bridge 逻辑里会多次复用
    """
    raise NotImplementedError("TODO: implement _add_reshape")
