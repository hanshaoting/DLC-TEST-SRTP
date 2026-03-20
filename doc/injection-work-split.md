# ONNX-Native Injection 分工方案

---

### 任务 1：Pattern Pool 加载与过滤

**负责文件：**

- `optmt/injection/pattern_pool.py`

**建议负责函数：**

- `PatternPool.load_all`
- `PatternPool._should_exclude`

**核心目标：**

完成 pattern library 的加载、过滤、建档与采样前准备工作。


**完成后产出：**

- 可加载有效 ONNX pattern
- 可正确过滤不适合注入的 pattern
- 可为后续 mutation 阶段提供可采样的 pattern 池

---

### 任务 2：ONNX 图操作原语

**负责文件：**

- `optmt/injection/onnx_utils.py`

**建议负责函数：**

- `prefix_graph`
- `get_tensor_info`
- `_extract_type_info`
- `collect_prefix_context`
- `topological_sort`
- `replace_input_references`
- `infer_and_check`
- `make_simple_model`

**核心目标：**

完成注入流程所需的底层 ONNX 图操作工具。


**完成后产出：**

- 能安全地给 pattern 子图加前缀
- 能从 ONNX 图中提取张量信息
- 能保证 graft 后图结构仍满足拓扑要求

---

### 任务 3：输入桥接 Input Bridge

**负责文件：**

- `optmt/injection/input_bridge.py`

**建议负责函数：**

- `plan_bridges`
- `_collect_available_tensors`
- `_resolve_target`
- `_find_exact_match`
- `_pick_source`
- `_build_bridge_chain`
- `_add_reshape`

**核心目标：**

实现 seed graph 到 pattern graph 的输入桥接，使 pattern 的自由输入可以从 anchor 前缀上下文中获得可兼容的输入。


**完成后产出：**

- 当存在完全匹配 tensor 时可直接复用
- 当 shape/dtype 不匹配时可生成桥接链
- bridge 后张量可通过 shape inference 且能在 ORT 中执行

---

### 任务 4：RPC 等价拼接策略

**负责文件：**

- `optmt/injection/onnx_rpc.py`

**建议负责函数：**

- `stitch`
- `_shape_align`
- `_zero_branch`
- `_inverse_cancel`
- `_concat_slice`
- `_align_for_concat`
- `_identity_residual`
- `_mk_reshape`
- `_np_dtype`

**核心目标：**

实现四种 ONNX-native RPC 策略，保证注入后的图在语义上与原图等价或近似等价。


**完成后产出：**

- 可根据策略构造出合法的 RPC 子图
- 在 ORT 中验证输出与 anchor 保持一致或近似一致

---

### 任务 5：注入总编排器 Mutator

**负责文件：**

- `optmt/injection/onnx_mutator.py`

**建议负责函数：**

- `OnnxMutator.mutate`
- `OnnxMutator.replay`
- `OnnxMutator._inject`
- `OnnxMutator._select_anchor`
- `OnnxMutator._find_pattern`

**核心目标：**

把前面所有模块串起来，完成一次完整的 ONNX-native injection 流程。


**完成后产出：**

- 可对 seed model 执行单次或多次 mutation
- 能记录 mutation record
- 能 replay 指定 mutation 序列
- 生成的 mutant model 可以通过 checker，并尽量保持 ORT 等价

---

## 三、任务依赖关系

推荐的依赖顺序如下：

1. **任务 1：Pattern Pool**
2. **任务 2：ONNX Utils**
3. **任务 3：Input Bridge**
4. **任务 4：ONNX RPC**
5. **任务 5：Onnx Mutator**

其中：

- 任务 3 依赖任务 2
- 任务 4 基本独立，但会复用一些 ONNX 图构造经验
- 任务 5 强依赖任务 1、2、3、4
