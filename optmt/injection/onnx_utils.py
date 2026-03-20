from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import onnx
from onnx import checker, helper, shape_inference


def prefix_graph(graph: onnx.GraphProto, prefix: str) -> onnx.GraphProto:
    """Namespace isolation: deep-copy graph and prefix all names.

    Prefixes should cover:
    - node.name
    - node.input[]
    - node.output[]
    - initializer.name
    - graph.input[].name
    - graph.output[].name
    - value_info[].name

    Returns a new GraphProto. The original graph must not be modified.

    任务 2：请完成该函数。

    实现目标：
    给一个 ONNX 子图做“命名空间隔离”，避免注入到 seed graph 后发生重名冲突。

    推荐实现步骤：
    1. 先对输入 `graph` 做深拷贝，确保原图不被修改。
    2. 建立一个 `rename` 映射：`旧 tensor 名 -> prefix + 旧 tensor 名`。
    3. 先处理这些显式声明位置：
       - `g.input`
       - `g.output`
       - `g.initializer`
       - `g.value_info`
    4. 再扫描所有 node 的 input/output，把还没加入映射的名字补进去。
       注意：空字符串输入（可选输入未使用）不要重命名。
    5. 最后统一回写：
       - `node.name` 也要加前缀
       - `node.input[i]` 和 `node.output[i]` 按 `rename` 替换
    6. 返回修改后的新图 `g`。

    易错点提示：
    - 不要原地修改传入的 `graph`
    - `node.name` 和 tensor name 不是一回事，但二者都建议加前缀
    - `initializer.name` 也属于 tensor 名的一部分，必须同步改
    - ONNX 里有些 input 可能是 `""`，表示可选输入缺失，不要处理它
    """
    raise NotImplementedError("TODO: implement prefix_graph")


def get_tensor_info(
    graph: onnx.GraphProto, tensor_name: str
) -> Optional[Tuple[List[int], int]]:
    """Query shape and elem_type of a tensor in the graph.

    Search order:
    1. value_info
    2. graph.input
    3. graph.output
    4. initializer

    Returns (shape_list, onnx_elem_type) or None if not found.

    任务 2：请完成该函数。

    目标：
    给定一个 tensor 名字，从图中查询它的 shape 和 dtype。

    推荐实现步骤：
    1. 按文档给定顺序依次查找：
       - `graph.value_info`
       - `graph.input`
       - `graph.output`
       - `graph.initializer`
    2. 对前三类 `ValueInfoProto`，可以调用你实现的 `_extract_type_info()`。
    3. 对 `initializer`：
       - shape 直接用 `list(init.dims)`
       - dtype 直接用 `init.data_type`
    4. 全部都找不到时返回 `None`。

    为什么查找顺序这样设计？
    - `value_info` 往往包含中间张量的推断结果
    - `graph.input` / `graph.output` 是图边界信息
    - `initializer` 是常量张量
    """
    raise NotImplementedError("TODO: implement get_tensor_info")


def _extract_type_info(
    value_info: onnx.ValueInfoProto,
) -> Optional[Tuple[List[int], int]]:
    """Extract shape and elem_type from a ValueInfoProto.

    Returns:
        (shape, elem_type) if available
        None if elem_type is missing / invalid

    任务 2：请完成该函数。

    推荐实现步骤：
    1. 取出 `value_info.type.tensor_type`
    2. 如果 `elem_type == 0`，说明类型未知，可直接返回 `None`
    3. 读取 shape：
       - 若存在 `shape` 字段，则遍历每个维度
       - 若 `dim.dim_value > 0`，加入真实数值
       - 否则加入 `-1`，表示 symbolic / unknown 维
    4. 返回 `(shape, elem_type)`

    说明：
    - 这里使用 `-1` 作为“未知维度”的统一占位
    - 后续模块若需要 concrete shape，会再自行把 `-1` 替换成默认值
    """
    raise NotImplementedError("TODO: implement _extract_type_info")


def collect_prefix_context(graph: onnx.GraphProto, anchor_name: str) -> Set[str]:
    """Collect all tensor names in the prefix context of anchor_name.

    The prefix context contains all tensors that are topologically
    before and including the node that produces `anchor_name`.
    Graph inputs should always be included. Initializers should also
    be included when appropriate.

    任务 2：请完成该函数。

    目标：
    收集 anchor 前缀上下文里的所有“可见 tensor”，供后续输入桥接阶段使用。

    推荐实现思路：
    1. 先建立 producer map：
       - `tensor_name -> producing node`
    2. 判断 `anchor_name` 是否有 producer：
       - 如果没有，说明 anchor 可能是 graph input 或 initializer
       - 此时可直接返回“图输入 + initializer”的集合
    3. 如果有 producer，则从 `anchor_name` 的生产节点开始，做一次“反向 BFS/DFS”：
       - 沿着当前节点的每个输入往前找它的 producer
       - 收集所有上游节点
    4. 最终把以下名字加入 context：
       - 所有 graph.input 的名字
       - 所有 initializer 的名字
       - 所有已访问节点的 output 名字
    5. 返回这个 `Set[str]`

    易错点：
    - 只需要收集“前缀上下文”，不要把下游 tensor 放进去
    - 可以用 `id(node)` 或索引来标识访问过的节点
    - 测试会检查像 diamond 图这种有分叉结构的情况
    """
    raise NotImplementedError("TODO: implement collect_prefix_context")


def topological_sort(graph: onnx.GraphProto) -> onnx.GraphProto:
    """Topologically sort graph nodes using Kahn's algorithm.

    ONNX requires node order to respect data dependencies.
    Return a new GraphProto whose node list is topologically sorted.

    任务 2：请完成该函数。

    目标：
    让图中的 node 顺序满足 ONNX checker 的拓扑要求。

    推荐实现思路（Kahn 算法）：
    1. 先收集“天然可用”的 tensor：
       - `graph.input`
       - `graph.initializer`
    2. 建立 `tensor_producer`：
       - `某个输出 tensor -> 生产它的 node 下标`
    3. 计算每个 node 的入度 `in_degree`：
       - 遍历 node 的每个输入
       - 如果该输入不是 graph input / initializer，
         且它由别的 node 生产，则当前 node 入度 +1
    4. 建立消费者表 `consumers`：
       - 某个 tensor 被哪些 node 消费
    5. 将所有 `in_degree == 0` 的节点入队
    6. 不断出队：
       - 追加到排序结果
       - 查看其输出被哪些消费者使用
       - 对这些消费者入度减 1
       - 若减到 0，则入队
    7. 若最后仍有未排序节点：
       - 说明图可能有环，或者依赖信息不完整
       - 为了尽量保持鲁棒性，可把剩余节点按原索引追加到末尾
    8. 复制原图，替换 node 顺序后返回

    注意：
    - 返回新图，不要直接修改输入图
    - `graph.node` 的其他字段不要丢失
    """
    raise NotImplementedError("TODO: implement topological_sort")


def replace_input_references(
    graph: onnx.GraphProto, old_name: str, new_name: str
) -> None:
    """Replace all node input references from `old_name` to `new_name`.

    Also update graph outputs if an output name equals `old_name`.
    This function operates in-place.

    任务 2：请完成该函数。

    目标：
    将图中所有“消费 old_name 的地方”改为消费 `new_name`。

    推荐实现步骤：
    1. 遍历 `graph.node`
    2. 对每个 node 的所有 input：
       - 若 `inp_name == old_name`，改成 `new_name`
    3. 再遍历 `graph.output`
       - 若某个输出名恰好等于 `old_name`，也改成 `new_name`

    说明：
    - 这里只替换“输入引用”，不要修改某个 node 的 output 名字
    - 这是注入阶段 reroute downstream consumers 的关键原语
    """
    raise NotImplementedError("TODO: implement replace_input_references")


def infer_and_check(model: onnx.ModelProto) -> onnx.ModelProto:
    """Run shape inference and ONNX checker, then return the inferred model.

    任务 2：请完成该函数。

    目标：
    为模型补全 shape/type 信息，并进行合法性校验。

    提示：
    - 可以先调用 `shape_inference.infer_shapes(model, check_type=True)`
    - 再调用 `checker.check_model(model)`
    - 最后返回推断后的 model

    这个函数实现很短，但在测试里很常用。
    """
    raise NotImplementedError("TODO: implement infer_and_check")


def make_simple_model(graph: onnx.GraphProto, opset: int = 17) -> onnx.ModelProto:
    """Wrap a GraphProto into a ModelProto with the given opset.

    The returned model should be shape-inferred before being returned.

    任务 2：请完成该函数。

    推荐实现步骤：
    1. 用 `helper.make_model()` 包装传入的 `graph`
    2. `opset_imports` 里放一个默认 domain (`\"\"`) 的 opset
    3. 调用 `shape_inference.infer_shapes(..., check_type=True)`
    4. 返回推断后的模型

    说明：
    - 这是测试中构造最小合法 ONNX 模型的便捷函数
    """
    raise NotImplementedError("TODO: implement make_simple_model")
