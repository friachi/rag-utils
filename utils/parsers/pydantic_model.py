from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


# -----------------------------
# public api
# -----------------------------

def parse_pydantic_model_to_markdown(python_text: str) -> str:
    """
    Output markdown with:
      - ## per top-level pydantic model (BaseModel subclass), ordered by references
      - ###/####... for nested classes (also BaseModel subclasses), under their parent
      - each section contains the class definition as python code (after comment stripping)

    Deterministic:
      - comment stripping via tokenize
      - reference ordering: topo sort, ties alphabetical, cycles broken alphabetical
    """
    cleaned = strip_python_comments(python_text)
    tree = ast.parse(cleaned)

    # index classes + nested structure
    index = build_class_index(tree)

    # keep only BaseModel classes
    base_models = {name for name, info in index.items() if info.is_basemodel}

    # top-level pydantic models (not nested under another class)
    top_level = sorted([name for name in base_models if index[name].parent is None])

    # dependency order among TOP-LEVEL models only
    deps = build_dependency_graph(index=index, model_names=set(top_level))
    ordered_top = topo_sort(names=top_level, deps=deps)

    # emit markdown
    lines: List[str] = []
    lines.append("# pydantic models")
    lines.append("")

    for cls_name in ordered_top:
        _emit_class_section(
            cls_name=cls_name,
            index=index,
            cleaned_source=cleaned,
            lines=lines,
            heading_level=2,
        )

    return ("\n".join(lines).strip() + "\n").lower()


# -----------------------------
# comment stripping
# -----------------------------

def strip_python_comments(source: str) -> str:
    """
    Remove all # comments deterministically using tokenize.
    Keeps code tokens; docstrings are not comments and remain, but are inside code fences anyway.
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
    parent: Optional[str]  # parent class name (top-level => None)
    children: Tuple[str, ...]  # nested class names
    is_basemodel: bool


def build_class_index(tree: ast.AST) -> Dict[str, ClassInfo]:
    """
    Index classes by a fully-qualified name:
      - top-level: "Foo"
      - nested: "Foo.Bar"
    """
    index: Dict[str, ClassInfo] = {}

    def walk_class(node: ast.ClassDef, parent_fqn: Optional[str]) -> str:
        fqn = node.name if parent_fqn is None else f"{parent_fqn}.{node.name}"

        # find nested classes
        child_fqns: List[str] = []
        for stmt in node.body:
            if isinstance(stmt, ast.ClassDef):
                child_fqns.append(walk_class(stmt, fqn))

        info = ClassInfo(
            node=node,
            parent=parent_fqn,
            children=tuple(child_fqns),
            is_basemodel=_is_basemodel_subclass(node),
        )
        index[fqn] = info
        return fqn

    if isinstance(tree, ast.Module):
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef):
                walk_class(stmt, None)

    return index


def _is_basemodel_subclass(cls: ast.ClassDef) -> bool:
    """
    True for:
      class X(BaseModel)
      class X(pydantic.BaseModel)
      class X(pydantic.main.BaseModel)
    Deterministic AST check (no imports executed).
    """
    for base in cls.bases:
        if isinstance(base, ast.Name) and base.id == "BaseModel":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseModel":
            return True
        if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Attribute) and base.attr == "BaseModel":
            return True
    return False


# -----------------------------
# dependency ordering
# -----------------------------

def build_dependency_graph(*, index: Dict[str, ClassInfo], model_names: Set[str]) -> Dict[str, Set[str]]:
    """
    Build dependencies among the given `model_names` only (typically top-level names).

    If A references B in any annotation inside A, then A depends on B (B should come first).
    Deterministic, conservative: identifier-based match on annotation text.
    """
    deps: Dict[str, Set[str]] = {n: set() for n in model_names}

    for name in model_names:
        node = index[name].node
        ann_texts = _collect_annotation_texts(node)
        for other in model_names:
            if other == name:
                continue
            if any(_contains_identifier(t, other) for t in ann_texts):
                deps[name].add(other)

    return deps


def topo_sort(*, names: List[str], deps: Dict[str, Set[str]]) -> List[str]:
    """
    Deterministic topo sort:
      - pick all "ready" nodes (no deps remaining) in alphabetical order each step
      - if cycle, break by picking the smallest remaining name
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
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            texts.append(_unparse(node.annotation))
        elif isinstance(node, ast.arg) and node.annotation is not None:
            texts.append(_unparse(node.annotation))
    return texts


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _contains_identifier(text: str, ident: str) -> bool:
    # exact identifier match: not part of a longer token
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
    if not info.is_basemodel:
        return  # only emit pydantic models

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
        # still deterministic: emit an empty fence if we can't extract
        lines.append("```python")
        lines.append(f"class {short_title}(BaseModel):")
        lines.append("    pass")
        lines.append("```")
        lines.append("")

    # nested classes as nested markdown sections
    # keep the nested order as in source (already preserved in `children`)
    for child_fqn in info.children:
        _emit_class_section(
            cls_name=child_fqn,
            index=index,
            cleaned_source=cleaned_source,
            lines=lines,
            heading_level=heading_level + 1,
        )
