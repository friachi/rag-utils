from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Optional, Union, Optional, Tuple
import ast
from pydantic import BaseModel, ConfigDict, Field

from utils.parsers.json_schema import parse_json_schema_to_markdown
from utils.parsers.json_instance import parse_json_instance_to_markdown
from utils.parsers.xsd_schema import parse_xsd_schema_to_markdown
from utils.parsers.xml_instance import parse_xml_instance_to_markdown
from utils.parsers.yaml_instance import parse_yaml_instance_to_markdown
from utils.parsers.pydantic_model import parse_pydantic_model_to_markdown
from utils.parsers.dataclass_model import parse_dataclass_model_to_markdown

class ConvertJsonRequest(BaseModel):
    filename: Optional[str] = None
    content_base64: str = Field(..., description="file content encoded as base64")

    max_example_length: Optional[int] = Field(default=None, ge=0)
    max_enum_values: Optional[int] = Field(default=None, ge=0)



class ConvertResponse(BaseModel):
    detected_type: str
    markdown: str


def decode_bytes_to_text(data: bytes) -> Tuple[str, str]:
    """
    Deterministic decode attempts in fixed order.
    Returns (text, encoding_used).
    Raises UnicodeDecodeError if all fail (latin-1 won't fail, so in practice it won't).
    """
    # note: latin-1 never fails, so this is effectively "always decodes".
    # if you want "hard fail" when not valid utf-* then remove latin-1.
    candidates = ["utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"]
    last_err: Optional[Exception] = None
    for enc in candidates:
        try:
            return data.decode(enc, errors="strict"), enc
        except Exception as e:
            last_err = e
    # should be unreachable with latin-1 included
    raise UnicodeDecodeError("unknown", b"", 0, 1, f"failed all decoders: {last_err}")



# -----------------------------
# Types
# -----------------------------

class FileNature(str, Enum):
    XSD_SCHEMA = "xsd_schema"
    XML_INSTANCE = "xml_instance"
    JSON_SCHEMA = "json_schema"
    JSON_INSTANCE = "json_instance"
    YAML_INSTANCE = "yaml_instance"
    PYDANTIC_MODEL = "pydantic_model"
    DATACLASS_MODEL = "dataclass_model"


class DetectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    nature: FileNature
    details: str


class ConvertOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: Optional[str] = None

    # new overrides (optional)
    max_example_length: Optional[int] = None
    max_enum_values: Optional[int] = None



# -----------------------------
# Public API
# -----------------------------

def convert_to_markdown(
    content: Union[str, bytes],
    *,
    filename: Optional[str] = None,
    options: Optional[ConvertOptions] = None,
) -> str:
    options = options or ConvertOptions(source_name=filename)
    text = _ensure_text(content)

    detection = detect_nature(text)

    # defaults (used only if user did not provide overrides)
    max_enum_values = options.max_enum_values if options.max_enum_values is not None else 20
    max_example_length = options.max_example_length if options.max_example_length is not None else 100

    if detection.nature == FileNature.XSD_SCHEMA:
        return parse_xsd_schema_to_markdown(text, max_enum_values=max_enum_values)

    if detection.nature == FileNature.XML_INSTANCE:
        return parse_xml_instance_to_markdown(text, max_example_length=max_example_length)

    if detection.nature == FileNature.JSON_SCHEMA:
        return parse_json_schema_to_markdown(text, max_enum_values=max_enum_values)

    if detection.nature == FileNature.JSON_INSTANCE:
        return parse_json_instance_to_markdown(text, max_example_length=max_example_length)

    if detection.nature == FileNature.YAML_INSTANCE:
        return parse_yaml_instance_to_markdown(text, max_example_length=max_example_length)

    if detection.nature == FileNature.PYDANTIC_MODEL:
        return parse_pydantic_model_to_markdown(text)

    if detection.nature == FileNature.DATACLASS_MODEL:
        return parse_dataclass_model_to_markdown(text)

    raise AssertionError("Unhandled FileNature")



# -----------------------------
# Deterministic detection
# -----------------------------

def detect_nature(text: str) -> DetectionResult:
    """
    Deterministic detection.
    """

    # Python models (must come first)
    py_model = _detect_python_model(text)
    if py_model is not None:
        return DetectionResult(
            nature=py_model,
            details=f"Python file defining {py_model.value}",
        )

    # XML (XSD XML)
    if _looks_like_xml(text):
        if _is_xsd_schema(text):
            return DetectionResult(
                nature=FileNature.XSD_SCHEMA,
                details="XML document containing XSD schema markers",
            )
        return DetectionResult(
            nature=FileNature.XML_INSTANCE,
            details="XML document",
        )

    # JSON (JSON Schema JSON)
    if _looks_like_json(text):
        if _is_json_schema(text):
            return DetectionResult(
                nature=FileNature.JSON_SCHEMA,
                details="JSON document containing JSON Schema markers",
            )
        return DetectionResult(
            nature=FileNature.JSON_INSTANCE,
            details="JSON document",
        )

    # YAML
    if _looks_like_yaml(text):
        return DetectionResult(
            nature=FileNature.YAML_INSTANCE,
            details="YAML document",
        )

    raise ValueError("Unable to deterministically detect file nature")


# -----------------------------
# Format checks
# -----------------------------

_XML_DECL_RE = re.compile(r"^\s*<\?xml\b", re.IGNORECASE)
_XML_TAG_RE = re.compile(r"^\s*<([A-Za-z_][\w\-.]*)(\s|>|/)", re.DOTALL)
_XSD_NS = "http://www.w3.org/2001/XMLSchema"


def _looks_like_xml(text: str) -> bool:
    if _XML_DECL_RE.search(text):
        return True
    return bool(_XML_TAG_RE.search(text))


def _is_xsd_schema(text: str) -> bool:
    """
    Deterministic XSD detection:
    True only when the *document root element* is an XSD <schema>.
    - <xs:schema ...> / <xsd:schema ...>
    - <schema xmlns="http://www.w3.org/2001/XMLSchema" ...>
    """
    # Find the first start tag after XML decl/comments/doctype.
    # This is not a full XML parser, but it's deterministic and good for detection.
    m = re.search(r"<\s*([A-Za-z_][\w\-.]*)(?::([A-Za-z_][\w\-.]*))?\b([^>]*)>", text, re.DOTALL)
    if not m:
        return False

    prefix = m.group(1)          # e.g. "xs" in <xs:schema>
    local = m.group(2) or m.group(1)  # e.g. "schema" (if no prefix)
    attrs = m.group(3) or ""

    # Case 1: prefixed schema: <xs:schema ...> or <xsd:schema ...>
    if m.group(2) is not None:
        # We have prefix:local
        if local.lower() != "schema":
            return False

        # Ensure that prefix is bound to the XSD namespace in this start tag.
        # e.g. xmlns:xs="http://www.w3.org/2001/XMLSchema"
        ns_decl = re.search(
            rf'xmlns\s*:\s*{re.escape(prefix)}\s*=\s*["\']{re.escape(_XSD_NS)}["\']',
            attrs,
            re.IGNORECASE,
        )
        return bool(ns_decl)

    # Case 2: unprefixed root: <schema xmlns="http://www.w3.org/2001/XMLSchema" ...>
    if local.lower() != "schema":
        return False

    default_ns = re.search(
        rf'xmlns\s*=\s*["\']{re.escape(_XSD_NS)}["\']',
        attrs,
        re.IGNORECASE,
    )
    return bool(default_ns)


def _looks_like_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def _is_json_schema(text: str) -> bool:
    obj = json.loads(text)
    if not isinstance(obj, dict):
        return False

    if "$schema" in obj:
        return True

    if obj.get("type") in {"object", "array"}:
        if any(k in obj for k in ("properties", "items", "required", "$defs", "definitions")):
            return True

    return False


def _looks_like_yaml(text: str) -> bool:
    if re.search(r"^\s*---\s*$", text, re.MULTILINE):
        return True
    if re.search(r"^\s*[\w\"'\-]+\s*:\s+.+$", text, re.MULTILINE):
        return True
    if re.search(r"^\s*-\s+.+$", text, re.MULTILINE):
        return True
    return False

def _detect_python_model(text: str) -> FileNature | None:
    """
    Detect whether a Python file defines:
    - pydantic.BaseModel subclasses
    - dataclasses (@dataclass)
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None

    has_pydantic = False
    has_dataclass = False

    for node in ast.walk(tree):
        # class Foo(BaseModel):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "BaseModel":
                    has_pydantic = True
                elif isinstance(base, ast.Attribute) and base.attr == "BaseModel":
                    has_pydantic = True

        # @dataclass
        if isinstance(node, ast.ClassDef):
            for deco in node.decorator_list:
                if isinstance(deco, ast.Name) and deco.id == "dataclass":
                    has_dataclass = True
                elif isinstance(deco, ast.Attribute) and deco.attr == "dataclass":
                    has_dataclass = True

    # deterministic precedence
    if has_pydantic:
        return FileNature.PYDANTIC_MODEL
    if has_dataclass:
        return FileNature.DATACLASS_MODEL

    return None


# -----------------------------
# Utilities
# -----------------------------

def _ensure_text(content: Union[str, bytes]) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, (bytes, bytearray)):
        text, _enc = decode_bytes_to_text(bytes(content))
        return text
    raise TypeError(f"Unsupported content type: {type(content)}")




from pathlib import Path
from typing import Tuple


def inspect_file(path: str | Path) -> tuple[DetectionResult, str]:
    """
    Load a file, return its detection result and raw content.

    This is intended for testing detection only.
    No parsing or conversion is performed.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(p)

    if not p.is_file():
        raise ValueError(f"Not a file: {p}")

    content = p.read_bytes()
    text = _ensure_text(content)

    detection = detect_nature(text)

    return detection, text

# detection, text = inspect_file(path='/Users/fahed/github/reranker-service/docs/Anna/ISIN/Foreign_Exchange/Foreign_Exchange.Forward.Non_Standard.InstRefDataReporting.V1.json')
# detection, text = inspect_file(path='/Users/fahed/github/reranker-service/docs/CDM_7.0.0/jsoncodelist/esma-product-classigication-1-0.json')
# detection, text = inspect_file(path='/Users/fahed/github/reranker-service/docs/FpML.5.1.13/fpml-asset-5-13.xsd')
# detection, text = inspect_file(path='/Users/fahed/github/reranker-service/docs/FpML.5.1.13/fx-derivatives/fx-ex01-fx-spot.xml')
# detection, text = inspect_file(path='/Users/fahed/github/reranker-service/docs/config.yaml')
# detection, text = inspect_file(path='/Users/fahed/github/reranker-service/docs/CDM_7.0.0/pydantic_model/cdm_product_collateral_CollateralInterestParameters_schema.py')
# detection, text = inspect_file(path='/Users/fahed/github/reranker-service/docs/CDM_7.0.0/pydantic_model/test.py')
# detection, text = inspect_file(path='/Users/fahed/github/reranker-service/docs/dataclass_sample.py')
# md = convert_to_markdown(text)
# print(md)


