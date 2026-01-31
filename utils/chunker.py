from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re
import math

from pydantic import BaseModel, Field
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter



class Chunk(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# -----------------------------
# Auto header-level selection
# -----------------------------
_MD_HEADER_RE = re.compile(r"(?m)^(?P<hashes>#{1,6})\s+(?P<title>.+?)\s*$")


def decide_best_max_header_level(
    markdown_text: str,
    *,
    max_chars: int = 6000,
    min_chunk_chars: int = 900,
    target_chunk_chars: Optional[int] = None,
    max_chunks_soft: int = 4000,
) -> int:
    """
    Decide a good max_header_level (1..6) by:
      - detecting deepest header nesting in the markdown
      - simulating header-based splitting at each level
      - scoring each level to avoid chunks that are too big or too small

    Returns: chosen max_header_level in [1..6]
    """
    levels = [len(m.group("hashes")) for m in _MD_HEADER_RE.finditer(markdown_text)]
    deepest = max(levels) if levels else 1
    deepest = max(1, min(6, deepest))

    if target_chunk_chars is None:
        target_chunk_chars = int(
            max(min_chunk_chars * 1.5, min(max_chars * 0.65, max_chars - 500))
        )

    def simulate_chunk_sizes(level: int) -> List[int]:
        matches = list(_MD_HEADER_RE.finditer(markdown_text))
        split_starts = [m.start() for m in matches if len(m.group("hashes")) <= level]

        if not split_starts:
            return [len(markdown_text)]

        starts = [0] if split_starts[0] != 0 else []
        starts += split_starts

        sizes: List[int] = []
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else len(markdown_text)
            sizes.append(max(0, e - s))
        return sizes

    def score(level: int, sizes: List[int]) -> float:
        if not sizes:
            return float("inf")

        n = len(sizes)
        big = sum(1 for x in sizes if x > max_chars)
        tiny = sum(1 for x in sizes if x < min_chunk_chars)

        sorted_sizes = sorted(sizes)
        median = sorted_sizes[n // 2]
        avg_abs_dev = sum(abs(x - target_chunk_chars) for x in sizes) / n

        big_rate = big / n
        tiny_rate = tiny / n

        chunk_count_penalty = 0.0
        if n > max_chunks_soft:
            chunk_count_penalty = math.log(n - max_chunks_soft + 1) * 10.0

        return (
            big_rate * 100.0
            + tiny_rate * 80.0
            + (avg_abs_dev / max(1.0, target_chunk_chars)) * 20.0
            + (abs(median - target_chunk_chars) / max(1.0, target_chunk_chars)) * 10.0
            + chunk_count_penalty
        )

    best_level = 1
    best_score = float("inf")
    for lvl in range(1, deepest + 1):
        sizes = simulate_chunk_sizes(lvl)
        s = score(lvl, sizes)
        if s < best_score:
            best_score = s
            best_level = lvl

    return best_level


def _build_headers_to_split_on(max_header_level: int) -> List[Tuple[str, str]]:
    if max_header_level < 1:
        raise ValueError("max_header_level must be >= 1 (or 0 to auto-select).")
    if max_header_level > 6:
        raise ValueError("Markdown only supports up to 6 header levels (# to ######).")
    return [(("#" * i), f"h{i}") for i in range(1, max_header_level + 1)]


# -----------------------------
# Core implementation (shared)
# -----------------------------
def _chunk_schema_markdown_core(
    markdown_text: str,
    *,
    source_file: Optional[str] = None,
    source_id: Optional[str] = None,
    # Structure
    max_header_level: int = 4,  # 1..6 or 0 for auto
    strip_headers: bool = False,
    # Size control
    max_chars: int = 6000,
    recursive_chunk_size: int = 2500,
    recursive_overlap: int = 200,
    recursive_separators: Optional[List[str]] = None,
    # Context preservation
    include_path_in_text: bool = True,
    path_line_prefix: str = "PATH: ",
    # Auto-selection tuning
    auto_min_chunk_chars: int = 900,
    auto_target_chunk_chars: Optional[int] = None,
) -> List[Chunk]:
    """
    Shared chunking logic for both file and text entrypoints.
    """
    if recursive_separators is None:
        recursive_separators = ["\n\n", "\n", " ", ""]

    chosen_level = max_header_level
    if max_header_level == 0:
        chosen_level = decide_best_max_header_level(
            markdown_text,
            max_chars=max_chars,
            min_chunk_chars=auto_min_chunk_chars,
            target_chunk_chars=auto_target_chunk_chars,
        )

    headers_to_split_on = _build_headers_to_split_on(chosen_level)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=strip_headers,
    )
    docs = header_splitter.split_text(markdown_text)

    header_labels_in_order = [label for _, label in headers_to_split_on]

    def build_header_path(meta: Dict[str, Any]) -> str:
        return " > ".join(
            str(meta[label]).strip()
            for label in header_labels_in_order
            if meta.get(label)
        )

    header_line_re = re.compile(r"^\s{0,3}#{1,6}\s+.*$")

    def split_leading_header_block(chunk_text: str) -> Tuple[str, str]:
        lines = chunk_text.splitlines()
        header_lines: List[str] = []
        i = 0
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        while i < len(lines) and header_line_re.match(lines[i]):
            header_lines.append(lines[i])
            i += 1
        return "\n".join(header_lines).strip(), "\n".join(lines[i:]).lstrip("\n")

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=recursive_chunk_size,
        chunk_overlap=recursive_overlap,
        separators=recursive_separators,
        length_function=len,
    )

    out: List[Chunk] = []

    for d in docs:
        meta = dict(d.metadata) if d.metadata else {}
        if source_file is not None:
            meta["source_file"] = source_file
        if source_id is not None:
            meta["source_id"] = source_id
        meta["chosen_max_header_level"] = chosen_level

        header_path = build_header_path(meta)
        meta["header_path"] = header_path

        path_prefix = (
            f"{path_line_prefix}{header_path}\n\n"
            if include_path_in_text and header_path
            else ""
        )

        content = d.page_content or ""

        if len(content) <= max_chars:
            out.append(Chunk(text=path_prefix + content, metadata=meta))
            continue

        header_block, remainder = split_leading_header_block(content)

        if not remainder.strip():
            out.append(Chunk(text=path_prefix + content, metadata=meta))
            continue

        sub_chunks = recursive_splitter.split_text(remainder)

        for i, sub in enumerate(sub_chunks):
            sub_meta = dict(meta)
            sub_meta["is_subchunk"] = True
            sub_meta["subchunk_index"] = i

            parts: List[str] = []
            if path_prefix:
                parts.append(path_prefix.rstrip())
            if header_block:
                parts.append(header_block)
            parts.append(sub.strip())

            out.append(
                Chunk(
                    text="\n\n".join(parts).strip() + "\n",
                    metadata=sub_meta,
                )
            )

    return out


# -----------------------------
# Public entrypoints
# -----------------------------
def chunk_schema_markdown_text(
    markdown: str,
    *,
    max_header_level: int,
    strip_headers: bool = False,
    max_chars: int = 6000,
    recursive_chunk_size: int = 2500,
    recursive_overlap: int = 200,
    recursive_separators: Optional[List[str]] = None,
    include_path_in_text: bool = True,
    path_line_prefix: str = "PATH: ",
    auto_min_chunk_chars: int = 900,
    auto_target_chunk_chars: Optional[int] = None,
    source_id: Optional[str] = None,
) -> List[Chunk]:
    """
    Chunk markdown from an input string.
    """
    return _chunk_schema_markdown_core(
        markdown,
        source_file=None,
        source_id=source_id,
        max_header_level=max_header_level,
        strip_headers=strip_headers,
        max_chars=max_chars,
        recursive_chunk_size=recursive_chunk_size,
        recursive_overlap=recursive_overlap,
        recursive_separators=recursive_separators,
        include_path_in_text=include_path_in_text,
        path_line_prefix=path_line_prefix,
        auto_min_chunk_chars=auto_min_chunk_chars,
        auto_target_chunk_chars=auto_target_chunk_chars,
    )


def chunk_schema_markdown_file(
    md_file: Union[str, Path],
    *,
    max_header_level: int,
    strip_headers: bool = False,
    max_chars: int = 6000,
    recursive_chunk_size: int = 2500,
    recursive_overlap: int = 200,
    recursive_separators: Optional[List[str]] = None,
    include_path_in_text: bool = True,
    path_line_prefix: str = "PATH: ",
    auto_min_chunk_chars: int = 900,
    auto_target_chunk_chars: Optional[int] = None,
    source_id: Optional[str] = None,
) -> List[Chunk]:
    """
    Chunk markdown from a file path.
    """
    md_file = Path(md_file)
    text = md_file.read_text(encoding="utf-8", errors="replace")

    return _chunk_schema_markdown_core(
        text,
        source_file=str(md_file),
        source_id=source_id or str(md_file),
        max_header_level=max_header_level,
        strip_headers=strip_headers,
        max_chars=max_chars,
        recursive_chunk_size=recursive_chunk_size,
        recursive_overlap=recursive_overlap,
        recursive_separators=recursive_separators,
        include_path_in_text=include_path_in_text,
        path_line_prefix=path_line_prefix,
        auto_min_chunk_chars=auto_min_chunk_chars,
        auto_target_chunk_chars=auto_target_chunk_chars,
    )


####### pydnatic model for api #######

class ChunkOut(BaseModel):
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkRequest(BaseModel):
    # Mandatory
    markdown: str = Field(..., description="Markdown content to chunk.")
    max_header_level: int = Field(
        ...,
        description=(
            "Header depth to split on. "
            "1 => '#', 2 => '#, ##', ... 6 => '#..######'. "
            "Use 0 to auto-select based on nesting + section sizes."
        ),
        ge=0,
        le=6,
        examples=[4],
    )

    # Optional (sensible defaults)
    strip_headers: bool = Field(
        False,
        description="Whether to strip headers from chunk text (recommended False for schema docs).",
    )

    # Size control
    max_chars: int = Field(
        6000,
        description="If a header-based chunk exceeds this many characters, it will be recursively split.",
        ge=200,
    )
    recursive_chunk_size: int = Field(
        2500,
        description="Recursive subchunk size in characters (only used when chunk > max_chars).",
        ge=100,
    )
    recursive_overlap: int = Field(
        200,
        description="Recursive subchunk overlap in characters.",
        ge=0,
    )
    recursive_separators: Optional[List[str]] = Field(
        None,
        description=(
            "Separators used for recursive splitting (highest priority first). "
            "If omitted, defaults to ['\\n\\n', '\\n', ' ', '']."
        ),
    )

    # Context preservation
    include_path_in_text: bool = Field(
        True,
        description="If True, prepends `PATH: <header_path>` to the top of each (sub)chunk.",
    )
    path_line_prefix: str = Field(
        "PATH: ",
        description="Prefix for the path line when include_path_in_text is True.",
        min_length=1,
        max_length=50,
    )

    # Auto-selection tuning (only used when max_header_level == 0)
    auto_min_chunk_chars: int = Field(
        900,
        description="Heuristic minimum chunk size when auto-selecting header depth.",
        ge=0,
    )
    auto_target_chunk_chars: Optional[int] = Field(
        None,
        description="Heuristic target chunk size when auto-selecting header depth (None = computed default).",
        ge=200,
    )

    # Metadata
    source_id: Optional[str] = Field(
        None,
        description="Optional source identifier to place into chunk metadata (e.g., filename, URI).",
    )


class ChunkResponse(BaseModel):
    chosen_max_header_level: int = Field(
        ...,
        description="The actual header level used (equals request.max_header_level unless request is 0).",
        ge=1,
        le=6,
    )
    chunks: List[ChunkOut]