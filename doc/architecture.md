# OPTMT Architecture — ONNX-Native Pipeline

> 更新时间: 2026-03-20

---

## 系统管线总览

```
SeedGenerator ──► OnnxMutator ──► DifferentialOracle ──► BugReport
  (NNSmith)    │   (inject)    │   (ORT vs TVM Relax) │   (reduce + attribute)
               │               │                      │
           seed.onnx    mutated.onnx              compare_metamorphic()
                                                  compare() (fallback)
```

---

## 目录结构 & 文件职责

### `optmt/seed/` — 种子生成

| 文件 | 职责 |
|------|------|
| `generator.py` | 封装 NNSmith 的 `model_gen()` 生成随机 ONNX 计算图；提供 `SeedConfig` 控制节点数/算子筛选；`make_random_input()` 为模型生成随机输入张量 |

### `optmt/injection/` — ONNX-Native 注入 (核心)

| 文件 | 职责 |
|------|------|
| `pattern_pool.py` | 扫描 `pattern_library/` 加载 ONNX pattern 文件；`_should_exclude()` 过滤量化/自定义域/非确定性算子 (Dropout, Random*)；提供 `sample()` 随机采样 `PatternEntry` |
| `onnx_utils.py` | ONNX 图操作原语：`get_tensor_info()` 查 shape/dtype、`prefix_graph()` 命名空间隔离、`replace_input_references()` 引用重写、`topological_sort()` 拓扑排序、`collect_prefix_context()` 收集 anchor 上游上下文 |
| `input_bridge.py` | OATest 式输入桥接：`plan_bridges()` 为 pattern 每个输入生成桥接计划 (Reuse 复用 / Bridge 广播对齐)；构造 ReduceSum→Reshape→Expand 节点链实现 shape 对齐 |
| `onnx_rpc.py` | 4 种 RPC (Reversibility-Preserving Connection) 策略的 ONNX 原生实现：`zero_branch` (x+P*0)、`inverse_cancel` (x+P-P)、`concat_slice` (Concat→Slice 恢复)、`identity_residual` (x+P*0 残差形式)；`stitch()` 统一入口生成等价拼接节点 |
| `onnx_mutator.py` | 7 步注入编排器 `OnnxMutator`：选 anchor→采样 pattern→命名空间隔离→输入桥接→RPC 拼接→节点嫁接→验证 (shape_inference + checker)。`mutate()` 支持多轮注入，内含 3 次 pattern 重试机制。`replay()` 支持确定性回放 |

### `optmt/oracle/` — 测试预言机

| 文件 | 职责 |
|------|------|
| `differential.py` | `DifferentialOracle`：`compare()` ORT vs TVM Relax 交叉后端比较；`compare_metamorphic()` 先验证 seed≈mutant on ORT (蜕变关系)，再 mutant ORT vs Relax (编译器 bug)；支持 NaN/Inf 检测 + atol+rtol 组合容差 |
| `attribution.py` | `attribute_bug()`：优化等级消融 (opt_level 0→4)，定位最低触发 bug 的优化等级 + 记录触发的 Pass 名称 |

### `optmt/probe/` — Pass 探针

| 文件 | 职责 |
|------|------|
| `pass_probe.py` | `PassProbe`：通过 `tvm.ir.instrument` 注册 Pass instrumentation，记录 TVM Relax 编译过程中触发的优化 Pass 名称列表 |

### `optmt/backend/` — 编译后端

| 文件 | 职责 |
|------|------|
| `tvm_relax.py` | `TVMRelax`：封装 ONNX→TVM Relax 导入 (`from_onnx`)、编译 (`build`)、执行 (`run`) 全流程；`_order_inputs_by_onnx()` 保证输入顺序与 ONNX 一致；支持可配 `opt_level` 和 PassProbe 集成 |

### `optmt/reducer/` — 缺陷归约

| 文件 | 职责 |
|------|------|
| `delta_debug.py` | `DeltaDebugger` (legacy GraphIR)、`OnnxDeltaDebugger` (ONNX-native)：经典 ddmin 算法最小化 mutation record 集合，找到触发 bug 的最小变异子集 |

### `optmt/` — 顶层编排

| 文件 | 职责 |
|------|------|
| `runner.py` | `FuzzRunner`：主 fuzzing 循环。`_fuzz_one_onnx()` 为 ONNX-native 路径：seed 生成→mutate (含 5 次 seed 重试)→metamorphic oracle→fallback oracle→bug 归约+保存。统计字段含 `total_seeds`, `total_mutations`, `bugs_found`, `bugs_reduced`, `errors`, `silent_skips` |
| `cli.py` | 命令行入口：`--onnx-native` 启用 ONNX-native 路径；支持 `--time-budget`, `--max-nodes`, `--max-mutations`, `-v` 等参数 |


