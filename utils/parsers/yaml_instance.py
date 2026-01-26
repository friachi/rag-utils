import json
import re
from typing import Any, Dict, List, Optional

import yaml  # requires pyyaml


# supported date formats (same as json/xml instance parsers):
# - yyyy-mm-dd
# - yyyy/mm/dd
# - yy-mm-dd
# - yy/mm/dd
_DATE_RE = re.compile(
    r"^(?:\d{4}|\d{2})[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])$"
)

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?$")


def parse_yaml_instance_to_markdown(
    yaml_text: str,
    *,
    max_example_length: int = 200,
    array_strategy: str = "first",  # "first" (sample first element) or "none"
) -> str:
    """
    Render a YAML instance into structured, all-lower-case markdown.

    Same rules as json/xml instance parsers:
      - headings mirror structure (#, ##, ###...)
      - bullets (max 2):
          - type: object|list|integer|float|date|string
          - example: actual value (when available)
      - deterministic:
          - mapping keys sorted alphabetically
          - lists: a single 'items' subsection; default sample first element
    """
    if max_example_length < 0:
        raise ValueError("max_example_length must be >= 0")
    if array_strategy not in {"first", "none"}:
        raise ValueError("array_strategy must be 'first' or 'none'")

    data = yaml.safe_load(yaml_text)

    lines: List[str] = []
    _render_yaml_node(
        name="yaml instance",
        value=data,
        lines=lines,
        heading_level=1,
        max_example_length=max_example_length,
        array_strategy=array_strategy,
    )

    return ("\n".join(lines).strip() + "\n").lower()


# -----------------------------
# rendering
# -----------------------------

def _render_yaml_node(
    *,
    name: str,
    value: Any,
    lines: List[str],
    heading_level: int,
    max_example_length: int,
    array_strategy: str,
) -> None:
    heading_level = max(1, min(6, heading_level))
    lines.append(f"{'#' * heading_level} {name}")

    t = _detect_yaml_type(value)
    bullets: List[str] = [f"- type: {t}"]

    ex = _example_for_yaml_value(value, max_example_length=max_example_length, array_strategy=array_strategy)
    if ex is not None and ex != "":
        bullets.append(f"- example: {ex}")

    lines.extend(bullets)
    lines.append("")

    # recurse
    if isinstance(value, dict):
        for k in sorted(value.keys(), key=lambda x: str(x)):
            _render_yaml_node(
                name=str(k),
                value=value[k],
                lines=lines,
                heading_level=heading_level + 1,
                max_example_length=max_example_length,
                array_strategy=array_strategy,
            )
    elif isinstance(value, list):
        if array_strategy == "first" and len(value) > 0:
            _render_yaml_node(
                name="items",
                value=value[0],
                lines=lines,
                heading_level=heading_level + 1,
                max_example_length=max_example_length,
                array_strategy=array_strategy,
            )
        # if "none", do not descend further


# -----------------------------
# type detection + examples
# -----------------------------

def _detect_yaml_type(value: Any) -> str:
    # containers
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "list"

    # yaml can decode ints/floats/bools/null directly
    if value is None or isinstance(value, bool):
        # not in allowed list; map deterministically
        return "string"

    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"

    # many yaml parsers can parse timestamps into date/datetime objects,
    # but we keep it simple and deterministic by using string pattern checks.
    if isinstance(value, str):
        s = value.strip()
        if _DATE_RE.match(s):
            return "date"
        # sometimes yaml values are numeric but quoted; keep detection consistent with json/xml
        if _INT_RE.match(s):
            return "integer"
        if _FLOAT_RE.match(s) and (("." in s) or ("e" in s.lower())):
            return "float"
        return "string"

    # fallback
    return "string"


def _example_for_yaml_value(value: Any, *, max_example_length: int, array_strategy: str) -> Optional[str]:
    if isinstance(value, dict):
        return None

    if isinstance(value, list):
        if len(value) == 0:
            return "[]"
        if array_strategy == "first":
            first = _example_for_yaml_value(value[0], max_example_length=max_example_length, array_strategy=array_strategy)
            if first is None:
                first = "{...}"
            return f"[{first}, ...]"
        return "[...]"

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)

    s = _one_line(str(value))
    if max_example_length == 0:
        return ""
    if len(s) > max_example_length:
        s = s[: max_example_length].rstrip() + "..."
    return s


def _one_line(s: str) -> str:
    return " ".join(str(s).replace("\r", " ").replace("\n", " ").split()).strip()
