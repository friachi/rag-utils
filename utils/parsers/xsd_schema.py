import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple


_XSD_NS = "http://www.w3.org/2001/XMLSchema"


# -----------------------------
# utils
# -----------------------------

def _ns(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag[1:].split("}", 1)[0]
    return ""


def _local(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _strip_prefix(qname: str) -> str:
    return qname.split(":", 1)[1] if ":" in qname else qname


def _clamp_h(level: int) -> int:
    return max(1, min(6, level))


def _one_line(s: str) -> str:
    return " ".join(str(s).replace("\r", " ").replace("\n", " ").split()).strip()


# -----------------------------
# bullets
# -----------------------------

def _bullets_for_xsd_node(node: ET.Element, *, kind: str, max_enum_values: int) -> List[str]:
    bullets: List[str] = []

    desc = _doc(node)
    if desc:
        bullets.append(f"- description: {desc}")

    t = _type_for_xsd_node(node)
    if t:
        bullets.append(f"- type: {t}")

    constraint = _constraints_for_xsd_node(node)
    if constraint:
        bullets.append(f"- constraint: {constraint}")

    required = _required_for_xsd_node(node)
    if required:
        bullets.append(f"- required: {required}")

    enums = _enums_for_xsd_node(node, max_enum_values=max_enum_values)
    if enums:
        bullets.append(f"- enums: {enums}")

    return bullets


def _doc(node: ET.Element) -> str:
    # xs:annotation/xs:documentation text (first found)
    for ann in node.findall(f"{{{_XSD_NS}}}annotation"):
        for doc in ann.findall(f"{{{_XSD_NS}}}documentation"):
            text = "".join(doc.itertext()).strip()
            if text:
                return _one_line(text)
    return ""


def _type_for_xsd_node(node: ET.Element) -> str:
    ln = _local(node.tag)

    if ln == "schema":
        tns = node.attrib.get("targetNamespace")
        return f"schema{(' ' + tns) if tns else ''}".strip()

    if ln in ("element", "attribute"):
        if "type" in node.attrib:
            return _strip_prefix(node.attrib["type"])
        if "ref" in node.attrib:
            return f"ref {_strip_prefix(node.attrib['ref'])}"
        # inline type
        for child in list(node):
            if _ns(child.tag) == _XSD_NS:
                cln = _local(child.tag)
                if cln == "complexType":
                    return "object"
                if cln == "simpleType":
                    return "string"
        return ""

    if ln == "complexType":
        return "object"
    if ln == "simpleType":
        # try to infer from restriction base
        rest = node.find(f"{{{_XSD_NS}}}restriction")
        if rest is not None and "base" in rest.attrib:
            return _strip_prefix(rest.attrib["base"])
        return "string"

    if ln in ("sequence", "choice", "all"):
        return ln
    if ln == "restriction":
        base = node.attrib.get("base")
        return _strip_prefix(base) if base else "restriction"
    if ln == "extension":
        base = node.attrib.get("base")
        return _strip_prefix(base) if base else "extension"

    return ""


def _required_for_xsd_node(node: ET.Element) -> str:
    ln = _local(node.tag)
    if ln == "element":
        # required if minOccurs absent or >= 1
        mino = node.attrib.get("minOccurs")
        if mino is None:
            return "true"
        try:
            return "true" if int(mino) >= 1 else "false"
        except Exception:
            return ""

    if ln == "attribute":
        use = node.attrib.get("use")
        if use in ("required", "optional", "prohibited"):
            return "true" if use == "required" else "false"
    return ""


def _constraints_for_xsd_node(node: ET.Element) -> str:
    parts: List[str] = []
    ln = _local(node.tag)

    # occurrences (elements inside particles)
    if ln == "element":
        if "minOccurs" in node.attrib:
            parts.append(f"minoccurs={node.attrib['minOccurs']}")
        if "maxOccurs" in node.attrib:
            parts.append(f"maxoccurs={node.attrib['maxOccurs']}")
        if "nillable" in node.attrib:
            parts.append(f"nillable={node.attrib['nillable']}")
        if "default" in node.attrib:
            parts.append(f"default={_one_line(node.attrib['default'])}")
        if "fixed" in node.attrib:
            parts.append(f"fixed={_one_line(node.attrib['fixed'])}")

    # attributes
    if ln == "attribute":
        if "use" in node.attrib:
            parts.append(f"use={node.attrib['use']}")
        if "default" in node.attrib:
            parts.append(f"default={_one_line(node.attrib['default'])}")
        if "fixed" in node.attrib:
            parts.append(f"fixed={_one_line(node.attrib['fixed'])}")

    # restriction facets (look under restriction anywhere directly below this node)
    rest = node.find(f"{{{_XSD_NS}}}restriction")
    if rest is None and ln == "restriction":
        rest = node

    if rest is not None:
        base = rest.attrib.get("base")
        if base:
            parts.append(f"base={_strip_prefix(base)}")

        facet_map = {
            "minInclusive": "minInclusive",
            "maxInclusive": "maxInclusive",
            "minExclusive": "minExclusive",
            "maxExclusive": "maxExclusive",
            "minLength": "minLength",
            "maxLength": "maxLength",
            "length": "length",
            "pattern": "pattern",
            "totalDigits": "totalDigits",
            "fractionDigits": "fractionDigits",
            "whiteSpace": "whiteSpace",
        }
        for child in list(rest):
            if _ns(child.tag) != _XSD_NS:
                continue
            fn = _local(child.tag)
            if fn in facet_map and "value" in child.attrib:
                parts.append(f"{facet_map[fn]}={_one_line(child.attrib['value'])}")

    return "; ".join(parts)


def _enums_for_xsd_node(node: ET.Element, *, max_enum_values: int) -> str:
    # gather enumeration values from restriction facets beneath this node
    rest = node.find(f"{{{_XSD_NS}}}restriction")
    if rest is None and _local(node.tag) == "restriction":
        rest = node
    if rest is None:
        return ""

    enums: List[str] = []
    for child in list(rest):
        if _ns(child.tag) == _XSD_NS and _local(child.tag) == "enumeration":
            v = child.attrib.get("value")
            if v is not None:
                enums.append(_one_line(v))

    if not enums:
        return ""

    total = len(enums)
    if max_enum_values == 0:
        return f"... (total: {total})"
    if total <= max_enum_values:
        return ", ".join(enums)

    head = enums[:max_enum_values]
    return f"{', '.join(head)}, ... (total: {total})"

def _render_xsd_simple_type(
    *,
    stype: ET.Element,
    lines: List[str],
    heading_level: int,
    heading_title: str,
    max_enum_values: int,
) -> None:
    lines.append(f"{'#' * _clamp_h(heading_level)} {heading_title}")

    bullets = _bullets_for_xsd_node(stype, kind="simpleType", max_enum_values=max_enum_values)
    if bullets:
        lines.extend(bullets)
        lines.append("")

    # restrictions list enums/patterns etc are already captured in bullets
    # but we still include nested restriction as structure if present
    for child in list(stype):
        if _ns(child.tag) != _XSD_NS:
            continue
        if _local(child.tag) in ("restriction", "list", "union"):
            lines.append(f"{'#' * _clamp_h(heading_level + 1)} {_local(child.tag)}")
            b = _bullets_for_xsd_node(child, kind=_local(child.tag), max_enum_values=max_enum_values)
            if b:
                lines.extend(b)
            lines.append("")

def parse_xsd_schema_to_markdown(xsd_text: str, *, max_enum_values: int = 20) -> str:
    """
    Parse an XSD (xml schema) into structured markdown.

    output rules (aligned with your json schema renderer):
    - headings (#, ##, ###...) mirror xsd structure (schema -> definitions -> nested particles)
    - under each heading, up to these bullets (only if available):
        - description:
        - type:
        - constraint:
        - required:
        - enums: (truncated deterministically to max_enum_values):
            enums: v1, v2, ... (total: n)
    - deterministic ordering:
        - global elements/types sorted by name
        - within compositors (sequence/choice/all), keep document order (deterministic for a given file)
    - final markdown is all-lower-case
    """
    if max_enum_values < 0:
        raise ValueError("max_enum_values must be >= 0")

    root = ET.fromstring(xsd_text)
    if _ns(root.tag) != _XSD_NS or _local(root.tag).lower() != "schema":
        raise ValueError("not an xsd schema: root element must be xsd <schema>")

    # collect global defs
    simple_types: Dict[str, ET.Element] = {}
    complex_types: Dict[str, ET.Element] = {}
    global_elements: Dict[str, ET.Element] = {}

    for child in list(root):
        if _ns(child.tag) != _XSD_NS:
            continue
        ln = _local(child.tag)
        name = child.attrib.get("name")
        if not name:
            continue
        if ln == "simpleType":
            simple_types[name] = child
        elif ln == "complexType":
            complex_types[name] = child
        elif ln == "element":
            global_elements[name] = child

    schema_title = (
        root.attrib.get("targetNamespace")
        or root.attrib.get("id")
        or root.attrib.get("version")
        or "xsd schema"
    )

    lines: List[str] = []
    lines.append(f"# {schema_title}")
    schema_bullets = _bullets_for_xsd_node(root, kind="schema", max_enum_values=max_enum_values)
    if schema_bullets:
        lines.extend(schema_bullets)
        lines.append("")

    # global elements
    if global_elements:
        lines.append("## elements")
        lines.append("")
        for name in sorted(global_elements.keys()):
            _render_xsd_element(
                elem=global_elements[name],
                lines=lines,
                heading_level=3,
                heading_title=name,
                simple_types=simple_types,
                complex_types=complex_types,
                max_enum_values=max_enum_values,
            )

    # global complex types
    if complex_types:
        lines.append("## complextypes")
        lines.append("")
        for name in sorted(complex_types.keys()):
            _render_xsd_complex_type(
                ctype=complex_types[name],
                lines=lines,
                heading_level=3,
                heading_title=name,
                simple_types=simple_types,
                complex_types=complex_types,
                max_enum_values=max_enum_values,
            )

    # global simple types
    if simple_types:
        lines.append("## simpletypes")
        lines.append("")
        for name in sorted(simple_types.keys()):
            _render_xsd_simple_type(
                stype=simple_types[name],
                lines=lines,
                heading_level=3,
                heading_title=name,
                max_enum_values=max_enum_values,
            )

    return ("\n".join(lines).strip() + "\n").lower()


# -----------------------------
# renderers
# -----------------------------

def _render_xsd_element(
    *,
    elem: ET.Element,
    lines: List[str],
    heading_level: int,
    heading_title: str,
    simple_types: Dict[str, ET.Element],
    complex_types: Dict[str, ET.Element],
    max_enum_values: int,
) -> None:
    lines.append(f"{'#' * _clamp_h(heading_level)} {heading_title}")

    bullets = _bullets_for_xsd_node(elem, kind="element", max_enum_values=max_enum_values)
    if bullets:
        lines.extend(bullets)
        lines.append("")

    # inline type
    for child in list(elem):
        if _ns(child.tag) != _XSD_NS:
            continue
        ln = _local(child.tag)
        if ln == "complexType":
            _render_xsd_complex_type(
                ctype=child,
                lines=lines,
                heading_level=heading_level + 1,
                heading_title="complextype",
                simple_types=simple_types,
                complex_types=complex_types,
                max_enum_values=max_enum_values,
            )
        elif ln == "simpleType":
            _render_xsd_simple_type(
                stype=child,
                lines=lines,
                heading_level=heading_level + 1,
                heading_title="simpletype",
                max_enum_values=max_enum_values,
            )

    # referenced named type
    tname = elem.attrib.get("type")
    if tname:
        qn = _strip_prefix(tname)
        if qn in complex_types:
            _render_xsd_complex_type(
                ctype=complex_types[qn],
                lines=lines,
                heading_level=heading_level + 1,
                heading_title=f"type: {qn}",
                simple_types=simple_types,
                complex_types=complex_types,
                max_enum_values=max_enum_values,
            )
        elif qn in simple_types:
            _render_xsd_simple_type(
                stype=simple_types[qn],
                lines=lines,
                heading_level=heading_level + 1,
                heading_title=f"type: {qn}",
                max_enum_values=max_enum_values,
            )

def _render_compositor(
    *,
    comp: ET.Element,
    lines: List[str],
    heading_level: int,
    heading_title: str,
    simple_types: Dict[str, ET.Element],
    complex_types: Dict[str, ET.Element],
    max_enum_values: int,
) -> None:
    lines.append(f"{'#' * _clamp_h(heading_level)} {heading_title}")

    bullets = _bullets_for_xsd_node(comp, kind=heading_title, max_enum_values=max_enum_values)
    if bullets:
        lines.extend(bullets)
        lines.append("")
    else:
        lines.append("")

    # document-order traversal for particles
    for child in list(comp):
        if _ns(child.tag) != _XSD_NS:
            continue
        ln = _local(child.tag)

        if ln == "element":
            name = child.attrib.get("name") or child.attrib.get("ref") or "element"
            _render_xsd_element(
                elem=child,
                lines=lines,
                heading_level=heading_level + 1,
                heading_title=_strip_prefix(name),
                simple_types=simple_types,
                complex_types=complex_types,
                max_enum_values=max_enum_values,
            )
        elif ln in ("sequence", "choice", "all"):
            _render_compositor(
                comp=child,
                lines=lines,
                heading_level=heading_level + 1,
                heading_title=ln,
                simple_types=simple_types,
                complex_types=complex_types,
                max_enum_values=max_enum_values,
            )
        elif ln == "any":
            lines.append(f"{'#' * _clamp_h(heading_level + 1)} any")
            b = _bullets_for_xsd_node(child, kind="any", max_enum_values=max_enum_values)
            if b:
                lines.extend(b)
            lines.append("")


def _render_extension_or_restriction(
    *,
    node: ET.Element,
    lines: List[str],
    heading_level: int,
    simple_types: Dict[str, ET.Element],
    complex_types: Dict[str, ET.Element],
    max_enum_values: int,
) -> None:
    kind = _local(node.tag)  # extension/restriction
    base = node.attrib.get("base")
    title = f"{kind}" + (f": {_strip_prefix(base)}" if base else "")
    lines.append(f"{'#' * _clamp_h(heading_level)} {title}")

    b = _bullets_for_xsd_node(node, kind=kind, max_enum_values=max_enum_values)
    if b:
        lines.extend(b)
        lines.append("")
    else:
        lines.append("")

    # elements/attributes inside extension/restriction
    for child in list(node):
        if _ns(child.tag) != _XSD_NS:
            continue
        ln = _local(child.tag)
        if ln in ("sequence", "choice", "all"):
            _render_compositor(
                comp=child,
                lines=lines,
                heading_level=heading_level + 1,
                heading_title=ln,
                simple_types=simple_types,
                complex_types=complex_types,
                max_enum_values=max_enum_values,
            )
        elif ln == "attribute":
            aname = child.attrib.get("name") or child.attrib.get("ref") or "attribute"
            _render_xsd_attribute(
                attr=child,
                lines=lines,
                heading_level=heading_level + 1,
                heading_title=f"@{_strip_prefix(aname)}",
                simple_types=simple_types,
                max_enum_values=max_enum_values,
            )


def _render_xsd_attribute(
    *,
    attr: ET.Element,
    lines: List[str],
    heading_level: int,
    heading_title: str,
    simple_types: Dict[str, ET.Element],
    max_enum_values: int,
) -> None:
    lines.append(f"{'#' * _clamp_h(heading_level)} {heading_title}")

    bullets = _bullets_for_xsd_node(attr, kind="attribute", max_enum_values=max_enum_values)
    if bullets:
        lines.extend(bullets)
        lines.append("")

    # inline simpleType
    for child in list(attr):
        if _ns(child.tag) == _XSD_NS and _local(child.tag) == "simpleType":
            _render_xsd_simple_type(
                stype=child,
                lines=lines,
                heading_level=heading_level + 1,
                heading_title="simpletype",
                max_enum_values=max_enum_values,
            )

    # referenced named simpleType
    tname = attr.attrib.get("type")
    if tname:
        qn = _strip_prefix(tname)
        if qn in simple_types:
            _render_xsd_simple_type(
                stype=simple_types[qn],
                lines=lines,
                heading_level=heading_level + 1,
                heading_title=f"type: {qn}",
                max_enum_values=max_enum_values,
            )


def _render_xsd_complex_type(
    *,
    ctype: ET.Element,
    lines: List[str],
    heading_level: int,
    heading_title: str,
    simple_types: Dict[str, ET.Element],
    complex_types: Dict[str, ET.Element],
    max_enum_values: int,
) -> None:
    lines.append(f"{'#' * _clamp_h(heading_level)} {heading_title}")

    bullets = _bullets_for_xsd_node(ctype, kind="complexType", max_enum_values=max_enum_values)
    if bullets:
        lines.extend(bullets)
        lines.append("")

    # content model: sequence/choice/all, complexContent, simpleContent
    for child in list(ctype):
        if _ns(child.tag) != _XSD_NS:
            continue
        ln = _local(child.tag)

        if ln in ("sequence", "choice", "all"):
            _render_compositor(
                comp=child,
                lines=lines,
                heading_level=heading_level + 1,
                heading_title=ln,
                simple_types=simple_types,
                complex_types=complex_types,
                max_enum_values=max_enum_values,
            )

        elif ln in ("complexContent", "simpleContent"):
            lines.append(f"{'#' * _clamp_h(heading_level + 1)} {ln}")
            lines.append("")
            # extension/restriction inside
            for cc in list(child):
                if _ns(cc.tag) != _XSD_NS:
                    continue
                if _local(cc.tag) in ("extension", "restriction"):
                    _render_extension_or_restriction(
                    
                        node=cc,
                        lines=lines,
                        heading_level=heading_level + 2,
                        simple_types=simple_types,
                        complex_types=complex_types,
                        max_enum_values=max_enum_values,
                    )

        elif ln == "attribute":
            aname = child.attrib.get("name") or child.attrib.get("ref") or "attribute"
            _render_xsd_attribute(
                attr=child,
                lines=lines,
                heading_level=heading_level + 1,
                heading_title=f"@{_strip_prefix(aname)}",
                simple_types=simple_types,
                max_enum_values=max_enum_values,
            )

        elif ln == "attributeGroup":
            gref = child.attrib.get("ref") or child.attrib.get("name") or "attributegroup"
            lines.append(f"{'#' * _clamp_h(heading_level + 1)} attributegroup: {_strip_prefix(gref)}")
            lines.append("")
