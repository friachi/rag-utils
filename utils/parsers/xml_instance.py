import re
import xml.etree.ElementTree as ET
from typing import Any, List, Optional


# supported date formats (same as json instance parser):
# - yyyy-mm-dd
# - yyyy/mm/dd
# - yy-mm-dd
# - yy/mm/dd
_DATE_RE = re.compile(
    r"^(?:\d{4}|\d{2})[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])$"
)

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][+-]?\d+)?$")


def parse_xml_instance_to_markdown(
    xml_text: str,
    *,
    max_example_length: int = 100,
) -> str:
    """
    Render an XML instance into structured, all-lower-case markdown.

    Each section includes up to two bullets:
      - type: object|list|integer|float|date|string
      - example: <actual value> (when available)

    Deterministic rules:
      - children elements are processed in document order
      - repeated sibling tags are represented as a 'list' section:
          <tag> appears multiple times => one section for 'tag' with type=list and an 'items' child
      - attributes are emitted as child sections under their owning element, named: @attr
    """
    if max_example_length < 0:
        raise ValueError("max_example_length must be >= 0")

    root = ET.fromstring(xml_text)
    lines: List[str] = []

    _render_xml_element(
        elem=root,
        lines=lines,
        heading_level=1,
        display_name=_local_name(root.tag),
        max_example_length=max_example_length,
    )

    return ("\n".join(lines).strip() + "\n").lower()


# -----------------------------
# rendering
# -----------------------------

def _render_xml_element(
    *,
    elem: ET.Element,
    lines: List[str],
    heading_level: int,
    display_name: str,
    max_example_length: int,
) -> None:
    heading_level = _clamp_h(heading_level)
    lines.append(f"{'#' * heading_level} {display_name}")

    # attributes as children (we still need type/example for the element first)
    child_elems = list(elem)

    # determine this element's "type"
    # - list: if this element has repeated child tags (handled at child grouping level, not here)
    # - object: if element has children or attributes
    # - otherwise infer from text
    if child_elems or elem.attrib:
        etype = "object"
        example = _example_for_text(_text_value(elem), max_example_length=max_example_length)
        # for object nodes, we include example only if there is meaningful text
        bullets = [f"- type: {etype}"]
        if example:
            bullets.append(f"- example: {example}")
    else:
        text = _text_value(elem)
        etype = _detect_scalar_type(text)
        example = _example_for_text(text, max_example_length=max_example_length)
        bullets = [f"- type: {etype}"]
        if example:
            bullets.append(f"- example: {example}")

    lines.extend(bullets)
    lines.append("")

    # render attributes
    for attr_name in sorted(elem.attrib.keys()):
        attr_value = elem.attrib.get(attr_name, "")
        _render_xml_attribute(
            name=attr_name,
            value=attr_value,
            lines=lines,
            heading_level=heading_level + 1,
            max_example_length=max_example_length,
        )

    # render children, grouping repeated sibling tags into list sections
    if child_elems:
        groups = _group_children_by_local_name_in_order(child_elems)
        for tag_name, children in groups:
            if len(children) == 1:
                _render_xml_element(
                    elem=children[0],
                    lines=lines,
                    heading_level=heading_level + 1,
                    display_name=tag_name,
                    max_example_length=max_example_length,
                )
            else:
                # repeated: represent as list
                _render_xml_list(
                    tag_name=tag_name,
                    elems=children,
                    lines=lines,
                    heading_level=heading_level + 1,
                    max_example_length=max_example_length,
                )


def _render_xml_list(
    *,
    tag_name: str,
    elems: List[ET.Element],
    lines: List[str],
    heading_level: int,
    max_example_length: int,
) -> None:
    heading_level = _clamp_h(heading_level)
    lines.append(f"{'#' * heading_level} {tag_name}")
    lines.append("- type: list")

    # example: show first item's text (if scalar) else "{...}"
    first = elems[0]
    if list(first) or first.attrib:
        ex = "{...}"
    else:
        ex = _example_for_text(_text_value(first), max_example_length=max_example_length) or ""
        if not ex:
            ex = "{...}"
    lines.append(f"- example: [{ex}, ...]")
    lines.append("")

    # items subsection (first element is a deterministic sample for structure)
    _render_xml_element(
        elem=first,
        lines=lines,
        heading_level=heading_level + 1,
        display_name="items",
        max_example_length=max_example_length,
    )


def _render_xml_attribute(
    *,
    name: str,
    value: str,
    lines: List[str],
    heading_level: int,
    max_example_length: int,
) -> None:
    heading_level = _clamp_h(heading_level)
    title = f"@{_local_name(name)}"
    lines.append(f"{'#' * heading_level} {title}")

    atype = _detect_scalar_type(value)
    ex = _example_for_text(value, max_example_length=max_example_length)

    lines.append(f"- type: {atype}")
    if ex:
        lines.append(f"- example: {ex}")
    lines.append("")


# -----------------------------
# type detection + examples
# -----------------------------

def _detect_scalar_type(text: str) -> str:
    """
    Deterministic type detection among:
      object, list, integer, float, date, string

    For xml element/attribute scalar content we return:
      integer|float|date|string
    """
    s = (text or "").strip()
    if not s:
        return "string"

    if _DATE_RE.match(s):
        return "date"
    if _INT_RE.match(s):
        return "integer"
    # float after int to avoid classifying ints as floats
    if _FLOAT_RE.match(s) and (("." in s) or ("e" in s.lower())):
        return "float"

    return "string"


def _example_for_text(text: str, *, max_example_length: int) -> str:
    if max_example_length == 0:
        return ""
    s = _one_line(text or "")
    if not s:
        return ""
    if len(s) > max_example_length:
        s = s[: max_example_length].rstrip() + "..."
    return s


def _text_value(elem: ET.Element) -> str:
    # concatenate direct text and tail-less inner texts only if no children?
    # for object nodes we only use elem.text (keeps it deterministic and compact)
    return elem.text or ""


# -----------------------------
# child grouping
# -----------------------------

def _group_children_by_local_name_in_order(children: List[ET.Element]) -> List[tuple[str, List[ET.Element]]]:
    """
    Group children by local-name, preserving first-seen order of tag groups.
    Example: a, b, a -> groups: [(a,[a1,a2]), (b,[b1])]
    """
    groups: List[tuple[str, List[ET.Element]]] = []
    index: dict[str, int] = {}

    for ch in children:
        name = _local_name(ch.tag)
        if name in index:
            groups[index[name]][1].append(ch)
        else:
            index[name] = len(groups)
            groups.append((name, [ch]))
    return groups


# -----------------------------
# utils
# -----------------------------

def _local_name(tag: str) -> str:
    # handles {ns}local and plain local
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _clamp_h(level: int) -> int:
    return max(1, min(6, level))


def _one_line(s: str) -> str:
    return " ".join(str(s).replace("\r", " ").replace("\n", " ").split()).strip()
