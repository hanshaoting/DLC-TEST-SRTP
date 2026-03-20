# Optimization-Aware Metamorphic Testing Framework for Deep Learning Compilers

---

## 1. Abstract

本框架提出一种面向深度学习编译器的 **优化感知蜕变测试 (Optimization-Aware Metamorphic Testing)** 方法。

**核心创新点**：

1. **职责分离的双层架构**：将"触发优化的 Pattern"与"保证等价的构造方法"严格解耦。Pattern 库仅负责生成能最大化触发编译器优化 Pass 的计算子图片段；语义等价性由独立的冗余路径构造策略 (Redundant Path Construction, RPC) 保证——拓展 MT-DLComp 仅有的单一 UOC 方式为多种构造策略。
2. **ONNX-Native 注入**：所有计算图突变（Pattern 拼接 + RPC 等价构造）直接在 ONNX protobuf 层完成。ONNX protobuf 是纯数据结构，无隐式图简化，所写即所得，彻底规避了中间转换吞掉冗余路径的风险。借鉴 OATest 论文的 Reuse/Bridge 策略实现跨图上下文融合。
3. **五阶管线闭环**：种子生成 → Pattern 库构建 → 等价注入变换 → 编译差分预言 → 缺陷归约上下文记录，全链路自动化。

---


## 2. Module Design: Stage-by-Stage

### 2.1 Stage 1: Seed Graph Generation

使用 NNSmith 通过 Z3 约束求解生成类型安全、形状合法的随机计算图作为初始种子。


### 2.2 Stage 2: Pattern Library Construction

**关键定位：Pattern 仅负责触发优化，不负责保证等价性。**

Pattern 是一段计算子图片段，其设计目标是：当被注入到种子图中时，能最大化地触发编译器内部特定优化 Pass 的执行。等价性由 Stage 3 的冗余路径构造策略独立保证。



### 2.3 Stage 3: ONNX-Native Equivalent Injection Transformation


#### 2.3.0 新数据流

种子图经 NNSmith 生成后，先物化为干净的 ONNX 模型（此时无冗余结构，export 不会破坏任何东西）。然后在 ONNX protobuf 层完成 7 步注入:

1. **锚点选择** — 在种子图中选一个中间张量（某节点的输出）作为注入点
2. **命名空间隔离** — 对 Pattern 图中所有名字加前缀 `p{index}_`
3. **输入桥接** — 借鉴 OATest 的 Reuse/Bridge 策略为 Pattern 自由输入寻找来源
4. **子图嫁接** — Pattern 的节点和权重（initializers）合并进种子图
5. **RPC 等价缝合** — 用 ONNX 原生算子将 Pattern 输出等价融回锚点
6. **下游重接** — 原消费 anchor 的节点改为消费 new_anchor
7. **合法性校验** — topological_sort + shape_inference + onnx.checker

#### 2.3.1 输入桥接 (借鉴 OATest)

Pattern 子图的自由输入需要合法的来源张量。OATest 提出两种策略:

**策略 A — Reuse**: 在 anchor 的前置上下文（拓扑序中位于 anchor 之前的所有张量）中寻找 shape 和 dtype 完全匹配的现有张量，直接作为 Pattern 输入。

**策略 B — Bridge**: 若无精确匹配，从前置上下文中随机选取一个张量作为 source，构造桥接算子链: Cast（dtype 对齐）→ Flatten → Slice/Pad（元素数对齐）→ Reshape（shape 对齐）。

关键洞察: 桥接链的数值不影响测试正确性 — RPC 策略保证 Pattern 输出最终被等价消除，所以截断/填充的具体数值无关紧要。

#### 2.3.2 RPC 策略族 (ONNX-Native)

MT-DLComp 仅有一种 UOC 构造方式。我们将其泛化为 4 种 RPC 策略，在 ONNX 算子层实现:

| 策略 | 数学不变量 | ONNX 算子链 | 优化触发面 |
|---|---|---|---|
| **RPC-0: Zero-Branch** | `x + P(in) × Guard(=0) = x` | Sub → Mul → Add(eps) → Neg → Relu → Mul → Add | MT-DLComp UOC 基线 |
| **RPC-1: Inverse-Cancel** | `x + P(in) - P(in) = x` | Add → Sub（同一张量名引用两次） | CSE、算术简化 |
| **RPC-2: Concat-Slice** | `Slice(Concat(x, P), 0:dim) = x` | Concat → Slice | Slice/Concat 融合 |
| **RPC-3: Identity-Residual** | `x × Sigmoid(1e7 + \|P\|) ≈ x` | Abs → ReduceSum → Add(1e7) → Sigmoid → Expand → Mul | 常量折叠、Sigmoid 融合 |

**通用 shape 对齐**: 当 P_out 与 anchor shape 不一致时，对 P_out 执行 ReduceSum(keepdims=0) → Reshape([1]) → Expand(anchor_shape)，将其压成标量再广播到 anchor shape。适用于 RPC-0/1/3 的兜底。RPC-2 对 concat axis 外的维度做 Flatten → Slice/Pad → Reshape 对齐。

RPC-3 仅在 float32/float64 下启用（IEEE 754 保证 `Sigmoid(x) = 1.0` 当 `x >= ~36.7`）。

#### 2.3.3 命名空间隔离

两张 ONNX 图合并时需处理 5 类命名冲突: 节点名、张量名、initializer 名、value_info 名、图级输入名。策略: 对 Pattern 图中所有名字加前缀 `p{mutation_index}_`，多次叠加注入时前缀递增。Pattern 的 graph.input（自由输入声明）不并入种子图的 graph.input — 它们被桥接策略替代。

#### 2.3.4 合法性校验
注入完成后执行三重校验:
1. 拓扑排序 — Kahn's algorithm 确保节点按依赖顺序排列 (ONNX spec 要求)
2. Shape inference — `onnx.shape_inference.infer_shapes` 填充所有中间张量类型
3. ONNX checker — `onnx.checker.check_model` 校验结构/类型/属性
4. (可选) ORT 烟雾测试 — 构造 `InferenceSession` 确认可执行

#### 2.3.5 注入策略对比 (v3.0 更新)

| 维度 | MT-DLComp | 本框架 (ONNX-Native) |
|---|---|---|
| **操作层** | ONNX protobuf | ONNX protobuf |
| **Pattern 职责** | Dead Code (隐含在 Guard 乘零中) | 仅触发优化，与等价性解耦 |
| **等价保证** | 仅 UOC (`x + dead×guard(=0)`) | 4 种 RPC 策略 |
| **上下文融合** | 无（Pattern 输入独立生成） | OATest 式 Reuse/Bridge |
| **Shape 匹配** | 手写 Slice/Pad/Unsqueeze (`node_gen.py`) | 通用 ReduceSum→Expand 兜底 + Bridge 链 |
| **可用 Pattern 数** | 固定 5 种 (Add/Sub/Mul/Conv/Dense) | ~320+ ONNX patterns + LLM 动态生成 |
| **RPC 结构保留** | 直接操作 ONNX | 直接操作 ONNX（所写即所得） |

---

### 2.4 Stage 4: Compilation & Differential Testing

#### 2.4.1 ONNX → TVM Relax: 单一路径 (v3.0 简化)

> **v3.0 更新**: ONNX-Native 注入后，路径 A/B 二选一的困境不复存在。
> 种子图的 `from_gir()` 导出不含冗余结构，不会被简化；所有冗余结构在 ONNX 层注入后直接送 TVM。

链路: Mutant ONNX (G') → `tvm.relax.frontend.onnx.from_onnx()` → Relax IR → `relax.build()` → VM

TVM `from_onnx()` 已验证为 1:1 直译（不做图简化），因此 ONNX 层注入的 RPC 结构在 Relax IR 中完整保留。路径 B（GraphIR → Relax 直连）不再需要开发。

#### 2.4.2 编译与执行


#### 2.4.3 差分预言机


**预言层级**:

| 症状 | 判定条件 | 严重级别 |
|---|---|---|
| Crash/Segfault | 编译或执行异常退出 | P0 |
| Compilation Error | `relax.build()` 抛异常 | P0 |
| Wrong Result | `max_abs_diff > 1e-4` | P1 |
| Precision Anomaly | `1e-5 < max_abs_diff <= 1e-4` | P2 |
| Timeout | 编译/执行超时 (300s) | P1 |

---

### 2.5 Stage 5: Bug Reduction & Context Logging

#### 2.5.1 三阶段归约 (借鉴 MT-DLComp `reduce/dd.py`)

**Phase 1: Pattern-Level Delta Debugging**

- 将注入历史视为 delta 序列
- 二分搜索最小触发 bug 的 pattern 子集
- 复用 MT-DLComp DeltaDebugging 算法 (`reduce/dd.py:73`)

**Phase 2: Node-Level Reduction**

- 在最小 pattern 子集内逐节点移除
- 保持 RPC 策略的等价不变量

**Phase 3: Pass Context Capture**

- 记录触发 bug 的完整 Pass 执行序列
- 二分 Pass Pipeline，定位最小触发 Pass 组合

#### 2.5.2 上下文记录输出

```python
@dataclass
class BugContext:
    seed_ir: GraphIR                    # 原始种子图
    mutant_ir: GraphIR                  # 突变图
    injected_patterns: List[Pattern]    # 注入的 Pattern 列表
    rpc_strategy: str                   # 使用的 RPC 策略
    trigger_passes: List[str]           # 触发 bug 的 Pass 序列
    min_pass_set: List[str]             # 最小 Pass 子集
    symptom: str                        # crash / wrong_result / timeout
    max_abs_diff: Optional[float]       # 数值偏差
    tvm_error_log: str                  # TVM 错误堆栈
    relax_ir_before_opt: str            # 优化前 Relax IR
    relax_ir_after_opt: str             # 优化后 Relax IR
    mre_script: str                     # 最小可复现 Python 脚本
```

---

## 3. Module Interoperability: 胶水适配层清单 (v3.0 更新)

| 适配层 | 位置 | 输入 → 输出 | 复杂度 |
|---|---|---|---|
| 种子图物化 | 复用 `nnsmith/materialize/onnx/` | `GraphIR` → `ONNX ModelProto` | 低 (现有链路) |
| ONNX-Native 注入 | `optmt/injection/onnx_mutator.py` | `Seed ONNX` → `Mutant ONNX` | 中 (Phase 5 核心) |
| 输入桥接 | `optmt/injection/input_bridge.py` | Pattern 自由输入 → 种子图张量 | 中 (Reuse + Bridge) |
| RPC 等价缝合 | `optmt/injection/onnx_rpc.py` | Pattern 输出 → 等价融回 anchor | 中 (4 种策略) |
| Pattern 池 | `optmt/injection/pattern_pool.py` | `pattern_library/*.onnx` → PatternEntry 列表 | 低 |
| ONNX → Relax | `tvm.relax.frontend.onnx.from_onnx()` | `ONNX ModelProto` → `tvm.IRModule` | 低 (TVM 内置) |
| Bug Report → TVM Issue | `reporter/tvm_issue.py` | `BugContext` → Markdown MRE | 低 |

---

## 4. Implementation Risks & Defensive Strategies


### Risk 1: 算子语义差异 (Lowering Gap)

**问题**: NNSmith AbsOp 语义与 Relax Op lowering 行为的微妙差异（Conv2d padding 语义、BatchNorm 返回值 tuple 等）。

**防御**: 每个算子映射附带单算子对拍测试，对拍基准为 NNSmith TorchModel eager 执行结果。

### Risk 2: RPC 策略的数值稳定性

**问题**: RPC-3 (Identity-Residual) 依赖 `Sigmoid(large_value) = 1.0` 的浮点精度假设，在低精度 (float16) 下可能不成立。

**防御**:

- RPC-3 仅在 float32/float64 下启用
- 每次注入后执行一次 eager 数值对拍 (reference run)，验证变换确实等价
- 若数值验证失败，自动 fallback 到 RPC-0 (Zero-Branch)

### Risk 3: TVM Pass Pipeline 版本依赖

**问题**: TVM Relax 快速迭代，Pass 名称/顺序/行为可能跨版本变化。

**防御**:

- 锁定 TVM commit hash
- Pass 名称通过配置文件引用，不硬编码
- 适配层加入 `tvm.__version__` 兼容性检查

### Risk 4: 注入后 ONNX 图拓扑/类型非法

**问题**: 在 ONNX protobuf 上注入 Pattern + Bridge + RPC 节点后可能出现: 节点顺序违反拓扑序、张量名悬空引用、shape inference 失败。

**防御**: 注入完成后执行三重校验:
1. `topological_sort` — Kahn's algorithm 重排节点
2. `onnx.shape_inference.infer_shapes` — 类型推导
3. `onnx.checker.check_model` — 结构/属性合法性
4. (可选) ORT `InferenceSession` 构造测试

---


## 5. Key File References

```
nnsmith/nnsmith/gir.py                       — GraphIR 核心数据结构
nnsmith/nnsmith/graph_gen.py                  — SymbolNet 种子图生成器
nnsmith/nnsmith/abstract/op.py                — ~60 个 AbsOpBase 算子定义
nnsmith/nnsmith/abstract/tensor.py            — AbsTensor 抽象张量
nnsmith/nnsmith/abstract/dtype.py             — DType 枚举
nnsmith/nnsmith/materialize/__init__.py        — Model 抽象基类 (from_gir 接口)
nnsmith/nnsmith/materialize/torch/forward.py  — Torch 算子映射 (Relax 映射参考模板)
nnsmith/nnsmith/backends/factory.py           — BackendFactory 抽象

Testing-DNN-Compilers/mutation/fcb_mut.py     — FCBMutator (UOC 注入参考)
Testing-DNN-Compilers/mutation/guard_gen.py   — UniversalGuard (Zero-Branch 等价保证)
Testing-DNN-Compilers/mutation/dead_gen.py    — Dead Code 生成 (Pattern 参考)
Testing-DNN-Compilers/mutation/node_gen.py    — Shape 匹配 (Slice/Pad 参考)
Testing-DNN-Compilers/reduce/dd.py            — Delta Debugging 算法
Testing-DNN-Compilers/compile/mass_compile.py — 批量编译执行参考

tvm/python/tvm/relax/op/                      — Relax 算子定义
tvm/python/tvm/relax/transform/               — Relax 优化 Pass
tvm/python/tvm/relax/build_module.py          — relax.build() 入口

OATest/                                       — OA 测试框架 (对比参考)
```
