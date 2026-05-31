"""Minimal prompt template primitives used by ScienceFlow prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InputField:
    name: str
    placeholder: str
    source: str = ""


@dataclass
class OutputField:
    name: str
    type: str


@dataclass
class PromptSection:
    title: str
    text: str
    title_level: int = 2

    def render(self, values: dict[str, str]) -> str:
        text = self.text
        for placeholder, value in values.items():
            text = text.replace(placeholder, value)
        return f"{'#' * self.title_level} {self.title}\n\n{text}".strip()


@dataclass
class PromptTemplate:
    name: str
    input_fields: list[InputField]
    output_key: str
    output_fields: list[OutputField]
    sections: list[PromptSection] = field(default_factory=list)
    requires_image: bool = False

    def render(self, payload: dict[str, Any]) -> str:
        values: dict[str, str] = {}
        for field in self.input_fields:
            value = payload.get(field.name)
            if isinstance(value, (dict, list)):
                rendered = json.dumps(value, ensure_ascii=False, indent=2)
            elif value is None:
                rendered = ""
            else:
                rendered = str(value)
            values[field.placeholder] = rendered
        return "\n\n".join(section.render(values) for section in self.sections)

