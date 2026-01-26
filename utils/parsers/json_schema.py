import json
from typing import Any, Dict, List, Optional, Set


def parse_json_schema_to_markdown(schema_text: str, *, max_enum_values: int = 5) -> str:
    """
    Fully parse a JSON Schema into structured markdown.

    - headings mirror structure (#, ##, ###...)
    - bullets only for info that exists
    - description bullet is first if present
    - required expanded
    - enums expanded up to max_enum_values, then truncated deterministically:
        enums: v1, v2, ... (total: n)
    - output is all-lower-case markdown
    """
    if max_enum_values < 0:
        raise ValueError("max_enum_values must be >= 0")

    schema = json.loads(schema_text)
    if not isinstance(schema, dict):
        raise ValueError("json schema root must be an object")

    lines: List[str] = []
    title = _pick_title(schema) or "json schema"

    _render_node(
        schema=schema,
        lines=lines,
        heading_level=1,
        heading_title=title,
        required_from_parent=None,
        property_name=None,
        max_enum_values=max_enum_values,
    )

    return ("\n".join(lines).strip() + "\n").lower()


# -----------------------------
# rendering
# -----------------------------

def _render_node(
    *,
    schema: Dict[str, Any],
    lines: List[str],
    heading_level: int,
    heading_title: str,
    required_from_parent: Optional[Set[str]],
    property_name: Optional[str],
    max_enum_values: int,
) -> None:
    heading_level = max(1, min(6, heading_level))
    lines.append(f"{'#' * heading_level} {heading_title}".strip())

    bullets = _bullets_for_section(
        schema=schema,
        required_from_parent=required_from_parent,
        property_name=property_name,
        max_enum_values=max_enum_values,
    )
    if bullets:
        lines.extend(bullets)
        lines.append("")

    # traverse structure
    defs = schema.get("$defs")
    if not isinstance(defs, dict):
        defs = schema.get("definitions")
    if isinstance(defs, dict) and defs:
        _render_map_container(
            name="$defs" if "$defs" in schema else "definitions",
            mapping=defs,
            lines=lines,
            heading_level=heading_level + 1,
            max_enum_values=max_enum_values,
        )

    props = schema.get("properties")
    if isinstance(props, dict) and props:
        required_list = schema.get("required")
        required_set = set(required_list) if isinstance(required_list, list) else set()
        for prop_name in sorted(props.keys()):
            prop_schema = props[prop_name]
            if isinstance(prop_schema, dict):
                _render_node(
                    schema=prop_schema,
                    lines=lines,
                    heading_level=heading_level + 1,
                    heading_title=prop_name,
                    required_from_parent=required_set,
                    property_name=prop_name,
                    max_enum_values=max_enum_values,
                )

    items = schema.get("items")
    if isinstance(items, dict):
        _render_node(
            schema=items,
            lines=lines,
            heading_level=heading_level + 1,
            heading_title="items",
            required_from_parent=None,
            property_name=None,
            max_enum_values=max_enum_values,
        )
    elif isinstance(items, list) and items:
        for i, it in enumerate(items):
            if isinstance(it, dict):
                _render_node(
                    schema=it,
                    lines=lines,
                    heading_level=heading_level + 1,
                    heading_title=f"items[{i}]",
                    required_from_parent=None,
                    property_name=None,
                    max_enum_values=max_enum_values,
                )

    addl = schema.get("additionalProperties")
    if isinstance(addl, dict):
        _render_node(
            schema=addl,
            lines=lines,
            heading_level=heading_level + 1,
            heading_title="additionalproperties",
            required_from_parent=None,
            property_name=None,
            max_enum_values=max_enum_values,
        )

    patt_props = schema.get("patternProperties")
    if isinstance(patt_props, dict) and patt_props:
        for patt in sorted(patt_props.keys()):
            sub = patt_props[patt]
            if isinstance(sub, dict):
                _render_node(
                    schema=sub,
                    lines=lines,
                    heading_level=heading_level + 1,
                    heading_title=f"patternproperties: {patt}",
                    required_from_parent=None,
                    property_name=None,
                    max_enum_values=max_enum_values,
                )

    prop_names = schema.get("propertyNames")
    if isinstance(prop_names, dict):
        _render_node(
            schema=prop_names,
            lines=lines,
            heading_level=heading_level + 1,
            heading_title="propertynames",
            required_from_parent=None,
            property_name=None,
            max_enum_values=max_enum_values,
        )

    contains = schema.get("contains")
    if isinstance(contains, dict):
        _render_node(
            schema=contains,
            lines=lines,
            heading_level=heading_level + 1,
            heading_title="contains",
            required_from_parent=None,
            property_name=None,
            max_enum_values=max_enum_values,
        )

    for key in ("allOf", "anyOf", "oneOf"):
        subs = schema.get(key)
        if isinstance(subs, list) and subs:
            _render_list_container(
                name=key.lower(),
                items=subs,
                lines=lines,
                heading_level=heading_level + 1,
                max_enum_values=max_enum_values,
            )

    neg = schema.get("not")
    if isinstance(neg, dict):
        _render_node(
            schema=neg,
            lines=lines,
            heading_level=heading_level + 1,
            heading_title="not",
            required_from_parent=None,
            property_name=None,
            max_enum_values=max_enum_values,
        )


def _render_map_container(
    *,
    name: str,
    mapping: Dict[str, Any],
    lines: List[str],
    heading_level: int,
    max_enum_values: int,
) -> None:
    heading_level = max(1, min(6, heading_level))
    lines.append(f"{'#' * heading_level} {name}")
    lines.append("")
    for k in sorted(mapping.keys()):
        v = mapping[k]
        if isinstance(v, dict):
            _render_node(
                schema=v,
                lines=lines,
                heading_level=heading_level + 1,
                heading_title=k,
                required_from_parent=None,
                property_name=None,
                max_enum_values=max_enum_values,
            )


def _render_list_container(
    *,
    name: str,
    items: List[Any],
    lines: List[str],
    heading_level: int,
    max_enum_values: int,
) -> None:
    heading_level = max(1, min(6, heading_level))
    lines.append(f"{'#' * heading_level} {name}")
    lines.append("")
    for i, it in enumerate(items):
        if isinstance(it, dict):
            _render_node(
                schema=it,
                lines=lines,
                heading_level=heading_level + 1,
                heading_title=f"{name}[{i}]",
                required_from_parent=None,
                property_name=None,
                max_enum_values=max_enum_values,
            )


# -----------------------------
# bullets formatting
# -----------------------------

def _bullets_for_section(
    *,
    schema: Dict[str, Any],
    required_from_parent: Optional[Set[str]],
    property_name: Optional[str],
    max_enum_values: int,
) -> List[str]:
    bullets: List[str] = []

    desc = schema.get("description")
    if isinstance(desc, str) and desc.strip():
        bullets.append(f"- description: {_one_line(desc)}")

    t = _infer_type(schema)
    if t is not None:
        bullets.append(f"- type: {t}")

    constraint_text = _constraints_text(schema)
    if constraint_text:
        bullets.append(f"- constraint: {constraint_text}")

    required_text = _required_text(schema, required_from_parent, property_name)
    if required_text:
        bullets.append(f"- required: {required_text}")

    enums_text = _enums_text(schema, max_enum_values=max_enum_values)
    if enums_text:
        bullets.append(f"- enums: {enums_text}")

    if bullets == ["- type: unknown"]:
        return []

    return bullets


def _infer_type(schema: Dict[str, Any]) -> Optional[str]:
    t = schema.get("type")
    if isinstance(t, str) and t.strip():
        return t.strip()
    if isinstance(t, list) and t:
        return " | ".join(sorted(str(x) for x in t))
    if "$ref" in schema and isinstance(schema["$ref"], str):
        return f"ref {schema['$ref']}"
    if "properties" in schema or "required" in schema:
        return "object"
    if "items" in schema:
        return "array"
    if "enum" in schema:
        return "enum"
    if "const" in schema:
        return "const"
    return None


def _constraints_text(schema: Dict[str, Any]) -> str:
    parts: List[str] = []

    if "$ref" in schema and isinstance(schema["$ref"], str):
        parts.append(f"$ref={schema['$ref']}")

    for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
        if k in schema:
            parts.append(f"{k}={schema[k]}")

    for k in ("minLength", "maxLength", "pattern", "format"):
        if k in schema:
            parts.append(f"{k}={_one_line(schema[k])}")

    for k in ("minItems", "maxItems"):
        if k in schema:
            parts.append(f"{k}={schema[k]}")
    if "uniqueItems" in schema:
        parts.append(f"uniqueItems={schema['uniqueItems']}")

    for k in ("minProperties", "maxProperties"):
        if k in schema:
            parts.append(f"{k}={schema[k]}")
    if "additionalProperties" in schema:
        ap = schema["additionalProperties"]
        if isinstance(ap, bool):
            parts.append(f"additionalProperties={ap}")
        elif isinstance(ap, dict):
            parts.append("additionalProperties=schema")

    if "default" in schema:
        parts.append(f"default={_one_line(schema['default'])}")

    return "; ".join(parts)


def _required_text(
    schema: Dict[str, Any],
    required_from_parent: Optional[Set[str]],
    property_name: Optional[str],
) -> str:
    if property_name is not None and required_from_parent is not None:
        return "true" if property_name in required_from_parent else "false"

    req = schema.get("required")
    if isinstance(req, list) and req:
        return ", ".join(sorted(str(x) for x in req))

    return ""


def _enums_text(schema: Dict[str, Any], *, max_enum_values: int) -> str:
    enum = schema.get("enum")
    if not (isinstance(enum, list) and enum):
        return ""

    # keep order as provided (deterministic)
    total = len(enum)

    if max_enum_values == 0:
        return f"... (total: {total})"

    if total <= max_enum_values:
        return ", ".join(_one_line(v) for v in enum)

    head = enum[:max_enum_values]
    head_text = ", ".join(_one_line(v) for v in head)
    return f"{head_text}, ... (total: {total})"


def _pick_title(schema: Dict[str, Any]) -> Optional[str]:
    for k in ("title", "$id", "id"):
        v = schema.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _one_line(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    return " ".join(s.replace("\r", " ").replace("\n", " ").split()).strip()

