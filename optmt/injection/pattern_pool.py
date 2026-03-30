from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import onnx

logger = logging.getLogger(__name__)

# Ops that indicate quantization — patterns containing these are excluded.
_QUANT_OPS = frozenset(
    {
        "QuantizeLinear",
        "DequantizeLinear",
        "DynamicQuantizeLinear",
        "MatMulInteger",
    }
)

# Nondeterministic ops — patterns containing these always fail RPC equivalence.
_NONDETERMINISTIC_OPS = frozenset(
    {
        "Dropout",
        "RandomUniform",
        "RandomNormal",
        "RandomNormalLike",
        "RandomUniformLike",
    }
)

# Custom domain prefix — patterns containing ops from this domain are excluded.
_CUSTOM_DOMAIN = "com.microsoft"


@dataclass
class PatternEntry:
    """A validated, ready-to-use ONNX pattern."""

    name: str
    category: str
    pass_name: str
    rel_path: str
    required_ops: List[str]
    graph: onnx.GraphProto
    opset: int
    # Metadata about inputs (from _pattern_metadata.json)
    input_info: List[Dict] = field(default_factory=list)
    output_names: List[str] = field(default_factory=list)


class PatternPool:
    """Load, filter, and serve ONNX patterns from pattern_library/."""

    def __init__(self, root: str = "pattern_library"):
        self.root = Path(root)
        self._entries: List[PatternEntry] = []
        self._by_category: Dict[str, List[PatternEntry]] = {}

    @property
    def entries(self) -> List[PatternEntry]:
        return self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def categories(self) -> Dict[str, int]:
        """Return {category: count} mapping."""
        return {cat: len(entries) for cat, entries in self._by_category.items()}

    def get_by_category(self, category: str) -> List[PatternEntry]:
        """Return all entries for a given category."""
        return self._by_category.get(category, [])

    def sample(
        self, k: int = 1, rng: Optional[random.Random] = None
    ) -> List[PatternEntry]:
        """Randomly sample k patterns from the pool."""
        r = rng or random.Random()
        return r.sample(self._entries, min(k, len(self._entries)))

    def load_all(self) -> int:
        """Load all valid patterns from pattern_library/."""
        # Clear existing entries to allow safe repeated calls
        self._entries.clear()
        self._by_category.clear()

        metadata_path = self.root / "_pattern_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        skipped_reasons: Dict[str, int] = {}
        loaded_count = 0

        for rel_path, info in metadata.items():
            # 1. Filter checks
            reason = self._should_exclude(rel_path, info)
            if reason:
                skipped_reasons[reason] = skipped_reasons.get(reason, 0) + 1
                continue

            # 2. File existence checks
            onnx_path = self.root / rel_path
            if not onnx_path.exists():
                skipped_reasons["file_missing"] = skipped_reasons.get("file_missing", 0) + 1
                continue

            # 3. Load and check model safely
            try:
                model = onnx.load(str(onnx_path), load_external_data=False)
                onnx.checker.check_model(model)
            except Exception as e:
                skipped_reasons["checker_fail"] = skipped_reasons.get("checker_fail", 0) + 1
                logger.debug(f"Failed to load or check model {rel_path}: {e}")
                continue

            # 4. Extract opset
            opset = 17
            if model.opset_import:
                for imp in model.opset_import:
                    if imp.domain == "":
                        opset = max(opset, imp.version)

            # 5. Extract metadata fields
            name = info.get("group", info.get("pass_name", "pattern"))
            category = info.get("pass_category", "unknown")
            pass_name = info.get("pass_name", "")

            req_ops = []
            for op_entry in info.get("required_ops", []):
                op_str = op_entry.get("op", "")
                op_name = op_str.split("::")[-1] if "::" in op_str else op_str
                req_ops.append(op_name)

            # 6. Construct PatternEntry
            entry = PatternEntry(
                name=name,
                category=category,
                pass_name=pass_name,
                rel_path=rel_path,
                required_ops=req_ops,
                graph=model.graph,
                opset=opset,
                input_info=info.get("input_info", []),
                output_names=info.get("output_names", [])
            )

            # 7. Store entry
            self._entries.append(entry)
            self._by_category.setdefault(category, []).append(entry)
            loaded_count += 1

        logger.info(f"Loaded {loaded_count} patterns. Skipped reasons: {skipped_reasons}")
        return loaded_count
        """Load all valid patterns from pattern_library/.

        任务 1：请完成该函数。

        推荐实现思路：
        1. 构造 metadata 文件路径：`self.root / "_pattern_metadata.json"`。
        2. 如果 metadata 不存在，抛出 `FileNotFoundError`。
        3. 读取 JSON metadata。
        4. 遍历 `metadata.items()`，其中：
           - key 是相对路径 `rel_path`
           - value 是该 pattern 的元信息 `info`
        5. 先调用 `_should_exclude(rel_path, info)`：
           - 如果返回非空 reason，则累计到 `skipped_reasons` 并跳过。
        6. 检查对应的 `.onnx` 文件是否存在：
           - 路径为 `self.root / rel_path`
           - 不存在时记录 `file_missing`
        7. 用 `onnx.load()` 读入模型，并用 `onnx.checker.check_model()` 做校验：
           - 如果读取/检查失败，记录 `checker_fail`
           - 注意：不要让单个坏 pattern 终止整个加载流程
        8. 提取 opset：
           - 只看空 domain (`domain == ""`) 的 opset_import
           - 取其中最大 version
           - 如果没有则默认 17
        9. 构造 `PatternEntry`：
           - `name` 优先 `group`，其次 `pass_name`，最后 `"pattern"`
           - `category` 默认 `"unknown"`
           - `pass_name` 默认空字符串
           - `required_ops` 需要从 metadata 的 `required_ops` 中提取纯 op 名
             例如 `"ai.onnx::Add"` -> `"Add"`
           - `graph` 存 `model.graph`
           - `input_info` 和 `output_names` 直接取 metadata 对应字段
        10. 把 entry 加入：
            - `self._entries`
            - `self._by_category.setdefault(...).append(...)`
        11. 记录日志并返回成功加载数量。

        额外提示：
        - 如果你希望 `load_all()` 可重复调用，记得思考是否需要先清空
          `self._entries` 和 `self._by_category`。
        - `required_ops` 在 metadata 里通常是“字典列表”，每个元素类似：
          `{"op": "ai.onnx::Relu", ...}`
        - 测试会检查：
          1. 加载数量
          2. 分类统计
          3. 采样结果可通过 ONNX checker
        """
        raise NotImplementedError("TODO: implement PatternPool.load_all")

    @staticmethod
    def _should_exclude(rel_path: str, info: Dict) -> Optional[str]:
        """Check if a pattern should be excluded. Returns reason or None."""
        if info.get("pass_category") == "qdq_optimization":
            return "qdq_category"

        for op_entry in info.get("required_ops", []):
            op_str = op_entry.get("op", "")
            
            if _CUSTOM_DOMAIN in op_str:
                return "custom_domain"
            
            if "::" in op_str:
                op_name = op_str.split("::")[-1]
            else:
                op_name = op_str

            if op_name in _QUANT_OPS:
                return "quant_op"
            
            if op_name in _NONDETERMINISTIC_OPS:
                return "nondeterministic_op"

        return None
        """Check if a pattern should be excluded. Returns reason or None.

        任务 1：请完成该函数。

        需要实现的过滤规则：
        1. 如果 `info.get("pass_category") == "qdq_optimization"`，
           返回 `"qdq_category"`。
        2. 遍历 `info.get("required_ops", [])` 中的每个 op entry：
           - 取出 `op_str = op_entry.get("op", "")`
           - 如果 `op_str` 包含 `_CUSTOM_DOMAIN`，返回 `"custom_domain"`
           - 提取纯 op 名：
             - 若有 `"::"`，取最后一段
             - 否则直接用原字符串
           - 如果纯 op 名在 `_QUANT_OPS` 中，返回 `"quant_op"`
           - 如果纯 op 名在 `_NONDETERMINISTIC_OPS` 中，
             返回 `"nondeterministic_op"`
        3. 都不命中时返回 `None`。

        设计意图说明：
        - `qdq_optimization` 属于量化相关 pass，本实验希望排除。
        - 自定义域 `com.microsoft` 可能引入 ONNX Runtime/TVM 兼容性问题。
        - `Dropout` / `Random*` 等非确定性算子会破坏“等价注入”假设。

        小提示：
        - `rel_path` 本身可以不用参与判断，但保留参数是为了后续日志扩展。
        - 该函数只负责“判断是否排除”，不要在这里直接打印日志。
        """
        raise NotImplementedError(
            "TODO: implement PatternPool._should_exclude"
        )
