from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple


# -----------------------------
# public api
# -----------------------------

def parse_dataclass_model_to_markdown(python_text: str) -> str:
    """
    dataclass parser:
      1) strip # comments
      2) ast-parse and extract dataclass classes (@dataclass)
      3) reorder top-level dataclasses by references between them
      4) emit markdown with nested sections; each section contains the class code as-is

    note: output is NOT lowercased to preserve python code.
    """
    cleaned = strip_python_comments(python_text)
    tree = ast.parse(cleaned)

    index = build_dataclass_index(tree)
    dataclasses_set = {name for name, info in index.items() if info.is_dataclass}

    top_level = sorted([name for name in dataclasses_set if index[name].parent is None])

    deps = build_dependency_graph(index=index, class_names=set(top_level))
    ordered_top = topo_sort(names=top_level, deps=deps)

    lines: List[str] = []
    lines.append("# dataclass models")
    lines.append("")

    for cls_name in ordered_top:
        _emit_class_section(
            cls_name=cls_name,
            index=index,
            cleaned_source=cleaned,
            lines=lines,
            heading_level=2,
        )

    return "\n".join(lines).strip() + "\n"


# -----------------------------
# comment stripping
# -----------------------------

def strip_python_comments(source: str) -> str:
    """
    Remove all # comments deterministically using tokenize.
    Docstrings remain (they are code, and will sit inside code fences).
    """
    out: List[tokenize.TokenInfo] = []
    reader = io.StringIO(source).readline
    for tok in tokenize.generate_tokens(reader):
        if tok.type == tokenize.COMMENT:
            continue
        out.append(tok)
    return tokenize.untokenize(out)


# -----------------------------
# class indexing
# -----------------------------

@dataclass(frozen=True)
class ClassInfo:
    node: ast.ClassDef
    parent: Optional[str]          # fully-qualified name of parent class, None if top-level
    children: Tuple[str, ...]      # fully-qualified names of nested classes
    is_dataclass: bool


def build_dataclass_index(tree: ast.AST) -> Dict[str, ClassInfo]:
    """
    Index classes by fully-qualified name:
      - top-level: "Foo"
      - nested: "Foo.Bar"
    """
    index: Dict[str, ClassInfo] = {}

    def walk_class(node: ast.ClassDef, parent_fqn: Optional[str]) -> str:
        fqn = node.name if parent_fqn is None else f"{parent_fqn}.{node.name}"

        child_fqns: List[str] = []
        for stmt in node.body:
            if isinstance(stmt, ast.ClassDef):
                child_fqns.append(walk_class(stmt, fqn))

        index[fqn] = ClassInfo(
            node=node,
            parent=parent_fqn,
            children=tuple(child_fqns),
            is_dataclass=_is_dataclass(node),
        )
        return fqn

    if isinstance(tree, ast.Module):
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                walk_class(stmt, None)

    return index


def _is_dataclass(cls: ast.ClassDef) -> bool:
    """
    True for:
      @dataclass
      @dataclasses.dataclass
      @dataclass(...)
      @dataclasses.dataclass(...)
    """
    for deco in cls.decorator_list:
        # @dataclass
        if isinstance(deco, ast.Name) and deco.id == "dataclass":
            return True

        # @dataclasses.dataclass
        if isinstance(deco, ast.Attribute) and deco.attr == "dataclass":
            return True

        # @dataclass(...)
        if isinstance(deco, ast.Call):
            fn = deco.func
            if isinstance(fn, ast.Name) and fn.id == "dataclass":
                return True
            if isinstance(fn, ast.Attribute) and fn.attr == "dataclass":
                return True

    return False


# -----------------------------
# dependency ordering
# -----------------------------

def build_dependency_graph(*, index: Dict[str, ClassInfo], class_names: Set[str]) -> Dict[str, Set[str]]:
    """
    Build dependencies among the given `class_names` only (typically top-level dataclasses).

    Rule:
      If A references B in any annotation inside A, then A depends on B (B should come first).

    Deterministic + conservative:
      identifier-based match on annotation text.
    """
    deps: Dict[str, Set[str]] = {n: set() for n in class_names}

    for name in class_names:
        node = index[name].node
        ann_texts = _collect_annotation_texts(node)
        for other in class_names:
            if other == name:
                continue
            if any(_contains_identifier(t, other) for t in ann_texts):
                deps[name].add(other)

    return deps


def topo_sort(*, names: List[str], deps: Dict[str, Set[str]]) -> List[str]:
    """
    Deterministic topo sort:
      - pick all ready nodes (no deps remaining) in alphabetical order
      - if a cycle exists, break it by choosing the smallest remaining name
    """
    remaining: Set[str] = set(names)
    ordered: List[str] = []

    while remaining:
        ready = sorted([n for n in remaining if not (deps.get(n, set()) & remaining)])
        if not ready:
            pick = sorted(remaining)[0]
            ordered.append(pick)
            remaining.remove(pick)
            continue
        for n in ready:
            ordered.append(n)
            remaining.remove(n)

    return ordered


def _collect_annotation_texts(cls: ast.ClassDef) -> List[str]:
    texts: List[str] = []
    for node in ast.walk(cls):
        # class fields
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            texts.append(_unparse(node.annotation))
        # __init__ args, etc.
        elif isinstance(node, ast.arg) and node.annotation is not None:
            texts.append(_unparse(node.annotation))
    return texts


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _contains_identifier(text: str, ident: str) -> bool:
    return bool(re.search(rf"(?<![\w]){re.escape(ident)}(?![\w])", text))


# -----------------------------
# markdown emission
# -----------------------------

def _emit_class_section(
    *,
    cls_name: str,
    index: Dict[str, ClassInfo],
    cleaned_source: str,
    lines: List[str],
    heading_level: int,
) -> None:
    info = index[cls_name]
    if not info.is_dataclass:
        return  # only emit dataclasses

    heading_level = max(1, min(6, heading_level))
    short_title = cls_name.split(".")[-1]
    lines.append(f"{'#' * heading_level} {short_title}")

    code = ast.get_source_segment(cleaned_source, info.node) or ""
    code = code.rstrip()

    if code:
        lines.append("```python")
        lines.append(code)
        lines.append("```")
        lines.append("")
    else:
        # deterministic fallback if extraction fails
        lines.append("```python")
        lines.append(f"@dataclass")
        lines.append(f"class {short_title}:")
        lines.append("    pass")
        lines.append("```")
        lines.append("")

    # nested classes stay nested under their parent, in source order
    for child_fqn in info.children:
        _emit_class_section(
            cls_name=child_fqn,
            index=index,
            cleaned_source=cleaned_source,
            lines=lines,
            heading_level=heading_level + 1,
        )
