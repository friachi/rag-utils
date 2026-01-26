import json
import re
from typing import Any, List, Optional


# supported date formats:
# - yyyy-mm-dd
# - yyyy/mm/dd
# - yy-mm-dd
# - yy/mm/dd
_DATE_RE = re.compile(
    r"^(?:\d{4}|\d{2})[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])$"
)


def parse_json_instance_to_markdown(
    json_text: str,
    *,
    max_example_length: int = 200,
    array_strategy: str = "first",  # "first" (sample first element) or "none"
) -> str:
    """
    Render a JSON instance into structured, all-lower-case markdown.

    Each section may include up to two bullets:
      - type: object|list|integer|float|date|string
      - example: <actual value> (when applicable)

    Deterministic rules:
      - object keys are sorted
      - lists: a single 'items' subsection; by default sample first element
    """
    if max_example_length < 0:
        raise ValueError("max_example_length must be >= 0")
    if array_strategy not in {"first", "none"}:
        raise ValueError("array_strategy must be 'first' or 'none'")

    data = json.loads(json_text)

    lines: List[str] = []
    _render_instance_node(
        name="json instance",
        value=data,
        lines=lines,
        heading_level=1,
        max_example_length=max_example_length,
        array_strategy=array_strategy,
    )

    return ("\n".join(lines).strip() + "\n").lower()


def _render_instance_node(
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

    t = _detect_value_type(value)
    bullets: List[str] = [f"- type: {t}"]

    ex = _example_for_value(value, max_example_length=max_example_length, array_strategy=array_strategy)
    if ex is not None and ex != "":
        bullets.append(f"- example: {ex}")

    # emit bullets only if they contain meaningful info (always at least type)
    lines.extend(bullets)
    lines.append("")

    # Recurse on objects/lists to recreate structure
    if isinstance(value, dict):
        for k in sorted(value.keys(), key=lambda x: str(x)):
            _render_instance_node(
                name=str(k),
                value=value[k],
                lines=lines,
                heading_level=heading_level + 1,
                max_example_length=max_example_length,
                array_strategy=array_strategy,
            )
    elif isinstance(value, list):
        if array_strategy == "first" and len(value) > 0:
            _render_instance_node(
                name="items",
                value=value[0],
                lines=lines,
                heading_level=heading_level + 1,
                max_example_length=max_example_length,
                array_strategy=array_strategy,
            )
        # if "none", do not descend further


def _detect_value_type(value: Any) -> str:
    # order matters: bool is subclass of int
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "list"

    if isinstance(value, bool) or value is None:
        # not in the allowed list; map deterministically
        return "string"

    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"

    if isinstance(value, str):
        s = value.strip()
        if _DATE_RE.match(s):
            return "date"
        return "string"

    return "string"


def _example_for_value(value: Any, *, max_example_length: int, array_strategy: str) -> Optional[str]:
    """
    Provide an example bullet when it makes sense.
    - dict: no example
    - list: compact preview
    - primitives: actual value
    """
    if isinstance(value, dict):
        return None

    if isinstance(value, list):
        if len(value) == 0:
            return "[]"
        if array_strategy == "first":
            first = _example_for_value(value[0], max_example_length=max_example_length, array_strategy=array_strategy)
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

    if isinstance(value, str):
        if max_example_length == 0:
            return ""
        s = _one_line(value)
        if len(s) > max_example_length:
            s = s[: max_example_length].rstrip() + "..."
        return s

    s = _one_line(str(value))
    if max_example_length and len(s) > max_example_length:
        s = s[: max_example_length].rstrip() + "..."
    return s


def _one_line(s: str) -> str:
    return " ".join(s.replace("\r", " ").replace("\n", " ").split()).strip()
