from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


@dataclass
class RPCResult:
    """Output of an RPC stitch operation."""

    nodes: List[onnx.NodeProto] = field(default_factory=list)
    initializers: List[onnx.TensorProto] = field(default_factory=list)
    new_anchor_name: str = ""


# ── public API ──────────────────────────────────────────────────

RPC_REGISTRY: Dict[str, str] = {
    "zero_branch": "RPC-0",
    "inverse_cancel": "RPC-1",
    "concat_slice": "RPC-2",
    "identity_residual": "RPC-3",
}


def stitch(
    strategy: str,
    anchor_name: str,
    anchor_shape: List[int],
    anchor_dtype: int,
    p_out_name: str,
    p_out_shape: List[int],
    prefix: str,
    *,
    concat_axis: int = 0,
) -> RPCResult:
    """Dispatch to the requested RPC strategy.

    任务 4：请完成该函数。

    目标：
    根据 `strategy` 分派到对应的 RPC 构造函数，并返回 `RPCResult`。

    需要支持的策略：
    - `"zero_branch"`      -> `_zero_branch(...)`
    - `"inverse_cancel"`   -> `_inverse_cancel(...)`
    - `"concat_slice"`     -> `_concat_slice(..., concat_axis=concat_axis)`
    - `"identity_residual"`-> `_identity_residual(...)`

    推荐实现步骤：
    1. 判断 `strategy` 的值
    2. 调用对应私有函数
    3. 若策略名未知，抛出 `ValueError(f"Unknown RPC strategy: {strategy}")`

    说明：
    - 这是整个 RPC 模块的统一入口
    - 外层 mutator 只依赖这个接口，不直接调用下面的私有实现
    """
    raise NotImplementedError("TODO: implement stitch")


# ── universal shape alignment shim ──────────────────────────────


def _shape_align(
    p_out: str,
    p_out_shape: List[int],
    anchor_shape: List[int],
    tag: str,
) -> Tuple[str, List[onnx.NodeProto], List[onnx.TensorProto]]:
    """ReduceSum -> Reshape([1]) -> Expand(anchor_shape).

    Reduces `p_out` to a scalar, then broadcasts it to `anchor_shape`.
    Returns `(aligned_name, nodes, initializers)`.

    任务 4：请完成该函数。

    目标：
    当 pattern 输出 shape 与 anchor shape 不一致时，把 `p_out` 对齐为和
    `anchor_shape` 一样的张量，以便后续 Add/Mul/Sub 等 RPC 操作可以直接使用。

    推荐实现步骤：
    1. 若 `p_out_shape == anchor_shape`：
       - 直接返回 `(p_out, [], [])`
    2. 否则构造三个阶段：
       a. `ReduceSum`
          - 对 `p_out` 的所有 axis 求和
          - `axes` 常量可以用 `np.arange(len(p_out_shape), dtype=np.int64)`
          - `keepdims=0`
       b. `Reshape`
          - 把 reduce 后的标量 reshape 成 `[1]`
       c. `Expand`
          - 再扩展到 `anchor_shape`
    3. 所有常量都要放进 `initializers`
    4. 返回最后对齐后的 tensor 名，以及新增节点/常量

    命名建议：
    - `axes_c = f"{tag}al_axes"`
    - `reduced = f"{tag}al_reduced"`
    - `rs_shape_c = f"{tag}al_rs_shape"`
    - `reshaped = f"{tag}al_rs"`
    - `expand_shape_c = f"{tag}al_expand_shape"`
    - `aligned = f"{tag}al_out"`

    易错点：
    - `axes` / `shape` 常量的 dtype 都应是 `np.int64`
    - `Expand` 的第二个输入是目标 shape 常量，不是属性
    """
    raise NotImplementedError("TODO: implement _shape_align")


# ── RPC-0: Zero-Branch ──────────────────────────────────────────


def _zero_branch(
    anchor_name: str,
    anchor_shape: List[int],
    anchor_dtype: int,
    p_out_name: str,
    p_out_shape: List[int],
    prefix: str,
) -> RPCResult:
    """anchor + aligned_P × Guard(≡0) = anchor.

    Guard construction idea:
        Sub(anchor, anchor) -> 0
        Mul(diff, diff) -> 0
        Add(0, eps) -> eps
        Neg(eps) -> negative
        Relu(negative) -> 0

    任务 4：请完成该函数。

    目标：
    构造一个严格等价于 `anchor` 的表达式：
        `anchor + aligned_P * 0`

    推荐实现步骤：
    1. 先设置：
       - `tag = f"{prefix}rpc0_"`
       - `nodes = []`
       - `inits = []`
    2. 调用 `_shape_align(...)` 把 `p_out_name` 对齐到 `anchor_shape`
    3. 构造 guard：
       a. `diff = Sub(anchor, anchor)`
       b. `sq = Mul(diff, diff)`
       c. `pos = Add(sq, eps)`
       d. `neg = Neg(pos)`
       e. `guard = Relu(neg)`  -> 恒等于 0
    4. 其中 `eps` 要做成 initializer：
       - 若 `anchor_dtype == TensorProto.FLOAT`，可用 `1e-7`
       - 否则可用更小值，如 `1e-15`
       - numpy dtype 可调用 `_np_dtype(anchor_dtype)`
    5. 再构造：
       - `zero_prod = Mul(aligned, guard)`
       - `new_anchor = Add(anchor_name, zero_prod)`
    6. 返回 `RPCResult(nodes=..., initializers=..., new_anchor_name=...)`

    说明：
    - 这个 RPC 的关键思想是“引入分支，但保证分支值恒为 0”
    - `aligned` 与 `guard` 最终应在 shape 上兼容
    """
    raise NotImplementedError("TODO: implement _zero_branch")


# ── RPC-1: Inverse-Cancel ──────────────────────────────────────


def _inverse_cancel(
    anchor_name: str,
    anchor_shape: List[int],
    anchor_dtype: int,
    p_out_name: str,
    p_out_shape: List[int],
    prefix: str,
) -> RPCResult:
    """anchor + aligned_P - aligned_P = anchor.

    任务 4：请完成该函数。

    目标：
    构造一个严格等价于 `anchor` 的表达式：
        `(anchor + aligned_P) - aligned_P`

    推荐实现步骤：
    1. 设置 `tag = f"{prefix}rpc1_"`
    2. 调用 `_shape_align(...)`
    3. 构造两个节点：
       - `added = Add(anchor_name, aligned)`
       - `new_anchor = Sub(added, aligned)`
    4. 返回 `RPCResult`

    提示：
    - 这里不需要复制 `aligned`，重复引用同一个 tensor 即可
    - 这是四种 RPC 中最直观的一种
    """
    raise NotImplementedError("TODO: implement _inverse_cancel")


# ── RPC-2: Concat-Slice ────────────────────────────────────────


def _concat_slice(
    anchor_name: str,
    anchor_shape: List[int],
    anchor_dtype: int,
    p_out_name: str,
    p_out_shape: List[int],
    prefix: str,
    *,
    concat_axis: int = 0,
) -> RPCResult:
    """Slice(Concat(anchor, aligned_P), 0:anchor_dim) = anchor.

    任务 4：请完成该函数。

    目标：
    先把 `anchor` 与 `aligned_P` 在某个轴上拼起来，再通过 Slice 只切回
    `anchor` 原本那一段，从而保证结果与原始 `anchor` 完全相同。

    推荐实现步骤：
    1. 设置 `tag = f"{prefix}rpc2_"`
    2. 先构造一个 `aligned_shape`：
       - 复制 `anchor_shape`
       - 仅把 `concat_axis` 位置改成 `1`
       - 表示：除了拼接轴外，其余维度都和 anchor 对齐
    3. 调用 `_align_for_concat(...)`
    4. 构造 `Concat([anchor_name, aligned], axis=concat_axis)`
    5. 构造 `Slice`，截取范围：
       - `start = 0`
       - `end = anchor_shape[concat_axis]`
       - `axis = concat_axis`
    6. `Slice` 所需 `starts/ends/axes` 都要放在 initializer 中
    7. 返回 `RPCResult`

    说明：
    - 这里的 shape 对齐规则与 `_shape_align()` 不同
    - `_shape_align()` 是“变成与 anchor 同 shape”
    - `_align_for_concat()` 是“除 concat 轴外对齐，concat 轴固定成 1”
    """
    raise NotImplementedError("TODO: implement _concat_slice")


def _align_for_concat(
    p_out: str,
    p_out_shape: List[int],
    target_shape: List[int],
    tag: str,
) -> Tuple[str, List[onnx.NodeProto], List[onnx.TensorProto]]:
    """Align `p_out` to `target_shape` for concat-based RPC.

    Strategy:
    - if shapes already match: direct reuse
    - if numel equal: one Reshape
    - else: Reshape(flat) -> Slice/Pad -> Reshape(target)

    任务 4：请完成该函数。

    目标：
    为 `concat_slice` 策略生成一个适合拼接的 `aligned_P`。

    推荐实现步骤：
    1. 若 `p_out_shape == target_shape`：
       - 直接返回 `(p_out, [], [])`
    2. 初始化：
       - `nodes = []`
       - `inits = []`
       - `cur = p_out`
       - `src_numel = math.prod(p_out_shape)`
       - `tgt_numel = math.prod(target_shape)`
    3. 若 `src_numel == tgt_numel`：
       - 调用 `_mk_reshape(cur, target_shape, tag, "ca_eq")`
    4. 否则：
       a. 先 flatten 到 `[src_numel]`
       b. 若 `src_numel > tgt_numel`：
          - 用 `Slice` 截断到前 `tgt_numel` 个元素
       c. 若 `src_numel < tgt_numel`：
          - 用 `Pad` 在尾部补零
       d. 再 reshape 到 `target_shape`
    5. 返回最终 tensor 名及新增节点/常量

    常量提示：
    - Slice 的 `starts=[0]`, `ends=[tgt_numel]`, `axes=[0]`
    - Pad 的 `pads=[0, tgt_numel - src_numel]`
    - 都应使用 `np.int64`

    说明：
    - 这里和 input bridge 的思路类似，但服务于 RPC-2
    """
    raise NotImplementedError("TODO: implement _align_for_concat")


# ── RPC-3: Identity-Residual ───────────────────────────────────


def _identity_residual(
    anchor_name: str,
    anchor_shape: List[int],
    anchor_dtype: int,
    p_out_name: str,
    p_out_shape: List[int],
    prefix: str,
) -> RPCResult:
    """anchor × Sigmoid(1e7 + ReduceSum(|P_out|)) ≈ anchor × 1.0.

    任务 4：请完成该函数。

    目标：
    构造一个“数值上极接近恒等”的残差缩放因子，使输出近似等于 `anchor`。

    参考结构：
        abs_out   = Abs(p_out)
        reduced   = ReduceSum(abs_out)
        biased    = Add(reduced, 1e7)
        saturated = Sigmoid(biased)     # 约等于 1
        rs_out    = Reshape(saturated, [1])
        expanded  = Expand(rs_out, anchor_shape)
        out       = Mul(anchor, expanded)

    推荐实现步骤：
    1. 设置 `tag = f"{prefix}rpc3_"`
    2. 构造 `Abs`
    3. 构造 `ReduceSum`：
       - axis 为 `np.arange(len(p_out_shape), dtype=np.int64)`
       - `keepdims=0`
    4. 构造一个值为 `1e7` 的常量 `big_c`
       - dtype 用 `_np_dtype(anchor_dtype)`
    5. `Add(reduced, big_c)` 后接 `Sigmoid`
    6. 将 sigmoid 结果 reshape 到 `[1]`
    7. 再 expand 到 `anchor_shape`
    8. 最后 `Mul(anchor_name, expanded)`
    9. 返回 `RPCResult`

    说明：
    - 这个策略依赖 sigmoid 饱和到 1
    - 它在数学上不一定“严格等于” anchor，但在 float32/float64 下通常足够接近
    - 对应测试一般会放宽到一个很小的 atol
    """
    raise NotImplementedError("TODO: implement _identity_residual")


# ── utilities ──────────────────────────────────────────────────


def _mk_reshape(
    inp: str,
    shape: List[int],
    tag: str,
    label: str,
) -> Tuple[str, List[onnx.NodeProto], List[onnx.TensorProto]]:
    """Create a Reshape node together with its shape initializer.

    Returns:
        (out_name, [reshape_node], [shape_initializer])

    任务 4：请完成该函数。

    推荐实现步骤：
    1. 构造输出名：
       - `out = f"{tag}{label}_rs"`
    2. 构造 shape 常量名：
       - `sc = f"{tag}{label}_shape"`
    3. 用 `numpy_helper.from_array(np.array(shape, dtype=np.int64), name=sc)`
       创建 shape initializer
    4. 用 `helper.make_node("Reshape", [inp, sc], [out], ...)`
       创建 Reshape 节点
    5. 返回 `(out, [node], [initializer])`

    说明：
    - 该函数会在 `_align_for_concat()` 中多次复用
    """
    raise NotImplementedError("TODO: implement _mk_reshape")


def _np_dtype(onnx_dtype: int) -> np.dtype:
    """Map ONNX floating dtype to NumPy dtype.

    任务 4：请完成该函数。

    要求：
    - `TensorProto.FLOAT`   -> `np.float32`
    - `TensorProto.DOUBLE`  -> `np.float64`
    - `TensorProto.FLOAT16` -> `np.float16`
    - 其他情况默认返回 `np.float32`

    推荐实现方式：
    1. 构造一个字典 `_MAP`
    2. `return _MAP.get(onnx_dtype, np.float32)`

    说明：
    - 该函数主要用于构造常量 initializer 时选择 numpy dtype
    """
    raise NotImplementedError("TODO: implement _np_dtype")
