# -*- coding: utf-8 -*-
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

# -------- Pydantic v1/v2 兼容层 --------
try:
    # pydantic v2
    from pydantic import ConfigDict
    _HAS_CONFIGDICT = True
except Exception:  # pragma: no cover
    ConfigDict = None
    _HAS_CONFIGDICT = False

try:
    # pydantic v2
    from pydantic import field_validator
    _HAS_FIELD_VALIDATOR = True
except Exception:  # pragma: no cover
    field_validator = None
    _HAS_FIELD_VALIDATOR = False

try:
    # pydantic v1
    from pydantic import validator
    _HAS_VALIDATOR = True
except Exception:  # pragma: no cover
    validator = None
    _HAS_VALIDATOR = False


class CompatModel(BaseModel):
    """
    统一提供：
    - model_dump / model_dump_json（pydantic v1 也能用）
    - .modeldump 属性（有些旧代码会直接读这个）
    - extra=allow（LLM 输出多字段时不炸）
    """
    if _HAS_CONFIGDICT:  # pydantic v2
        model_config = ConfigDict(extra="allow", populate_by_name=True)

    else:  # pydantic v1
        class Config:
            extra = "allow"
            allow_population_by_field_name = True

        # v1 没有 model_dump / model_dump_json：补齐
        def model_dump(self, *args, **kwargs):
            return self.dict(*args, **kwargs)

        def model_dump_json(self, *args, **kwargs):
            return self.json(*args, **kwargs)

    @property
    def modeldump(self) -> Dict[str, Any]:
        # 兼容某些旧日志/调试代码使用 .modeldump（而不是 .model_dump()）
        return self.model_dump()


class Node(CompatModel):
    label: str = Field(
        description="Nodes label (e.g., 'Person', 'Location', 'Object', 'Value', 'Date', 'Concept', 'Event', 'TextUnit')."
    )
    name: str = Field(
        description="Core name or identifier. For numbers or dates, use the specific value (e.g., '3.83', 'March 15th')."
    )
    properties: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata including 'description', 'unit', 'confidence' and any specific attributes.",
    )


class Relationship(CompatModel):
    source_node_name: str = Field(
        default="",
        description="Name of the source node.",
    )
    source_node_label: str = Field(
        default="Concept",
        description="Label of the source node.",
    )
    target_node_name: str = Field(
        default="",
        description="Name of the target node.",
    )
    target_node_label: str = Field(
        default="Concept",
        description="Label of the target node.",
    )

    type: str = Field(description="Type of relation (e.g., 'COSTS', 'MEASURES', 'BASED_ON', 'IS_A').")
    properties: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata including 'source_of_belief', 'confidence' (0.0-1.0), and 'is_lie'.",
    )

    # ---------- 输入兼容：把旧字段映射到新字段 ----------
    if _HAS_FIELD_VALIDATOR:  # pydantic v2
        from pydantic import model_validator

        @model_validator(mode="before")
        @classmethod
        def _compat_keys(cls, data):
            if not isinstance(data, dict):
                return data

            # 如果新键缺失，则从旧键补齐
            data = dict(data)
            data.setdefault("source_node_name", data.get("sourcenodename"))
            data.setdefault("source_node_label", data.get("sourcenodelabel"))
            data.setdefault("target_node_name", data.get("targetnodename"))
            data.setdefault("target_node_label", data.get("targetnodelabel"))

            return data

        @field_validator("type")
        @classmethod
        def _upper_type(cls, v: str):
            return (v or "").strip().upper()

    elif _HAS_VALIDATOR:  # pydantic v1
        from pydantic import root_validator

        @root_validator(pre=True)
        def _compat_keys(cls, values):
            if not isinstance(values, dict):
                return values

            values = dict(values)
            values.setdefault("source_node_name", values.get("sourcenodename"))
            values.setdefault("source_node_label", values.get("sourcenodelabel"))
            values.setdefault("target_node_name", values.get("targetnodename"))
            values.setdefault("target_node_label", values.get("targetnodelabel"))
            return values

        @validator("type")
        def _upper_type(cls, v: str):
            return (v or "").strip().upper()

    # ---------- 访问兼容：旧属性名 -> 新属性名 ----------
    @property
    def sourcenodename(self) -> str:
        return self.source_node_name

    @property
    def sourcenodelabel(self) -> str:
        return self.source_node_label

    @property
    def targetnodename(self) -> str:
        return self.target_node_name

    @property
    def targetnodelabel(self) -> str:
        return self.target_node_label


class KnowledgeGraphExtraction(CompatModel):
    nodes: List[Node] = Field(description="List of all extracted nodes.")
    relationships: List[Relationship] = Field(description="List of all relationships.")
