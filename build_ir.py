#!/usr/bin/env python3
"""
IR Builder driven by bundle.yaml (YAML supports comments).

Bundle folder must contain:
  - bundle.yaml
  - schemas/ (xsd and/or json schema)
  - examples/ (optional: xml/json)

Outputs (in bundle folder by default):
  - <name>_ir.json
  - <name>_mappings.json (if mappings.enabled)

Usage:
  python build_ir.py /path/to/bundle
  python build_ir.py /path/to/bundle --out /path/to/outdir
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import yaml
from lxml import etree


# -------------------------
# IR datamodel (minimal)
# -------------------------

Json = Dict[str, Any]


def _nowish_note() -> str:
    return "generated-by-build_ir"


@dataclass
class Provenance:
    source_id: str
    version: Optional[str] = None
    locator: Optional[str] = None
    snippet: Optional[str] = None
    note: Optional[str] = None
    confidence: Optional[float] = None

    def to_json(self) -> Json:
        d = {"source_id": self.source_id}
        if self.version:
            d["version"] = self.version
        if self.locator:
            d["locator"] = self.locator
        if self.snippet:
            d["snippet"] = self.snippet[:2000]
        if self.note:
            d["note"] = self.note
        if self.confidence is not None:
            d["confidence"] = float(self.confidence)
        return d


@dataclass
class Cardinality:
    min: int = 0
    max: Optional[int] = None  # None => unbounded

    def to_json(self) -> Json:
        return {"min": self.min, "max": self.max}


@dataclass
class Constraint:
    kind: str
    id: Optional[str] = None
    target: Optional[str] = None
    values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    minLength: Optional[int] = None
    maxLength: Optional[int] = None
    format: Optional[str] = None
    const: Any = None
    rule: Optional[str] = None
    message: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Provenance] = field(default_factory=list)

    def to_json(self) -> Json:
        d: Json = {"kind": self.kind}
        if self.id:
            d["id"] = self.id
        if self.target:
            d["target"] = self.target
        if self.values is not None:
            d["values"] = self.values
        if self.pattern:
            d["pattern"] = self.pattern
        if self.minimum is not None:
            d["minimum"] = self.minimum
        if self.maximum is not None:
            d["maximum"] = self.maximum
        if self.minLength is not None:
            d["minLength"] = self.minLength
        if self.maxLength is not None:
            d["maxLength"] = self.maxLength
        if self.format:
            d["format"] = self.format
        if self.const is not None:
            d["const"] = self.const
        if self.rule:
            d["rule"] = self.rule
        if self.message:
            d["message"] = self.message
        if self.annotations:
            d["annotations"] = self.annotations
        if self.provenance:
            d["provenance"] = [p.to_json() for p in self.provenance]
        return d


@dataclass
class FieldIR:
    name: str
    type: Union[str, Dict[str, Any]]  # primitive string or {"ref": "..."}
    description: Optional[str] = None
    cardinality: Optional[Cardinality] = None
    required: Optional[bool] = None
    nullable: Optional[bool] = None
    default: Any = None
    examples: List[Any] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Provenance] = field(default_factory=list)

    def to_json(self) -> Json:
        d: Json = {"name": self.name, "type": self.type}
        if self.description:
            d["description"] = self.description
        if self.cardinality:
            d["cardinality"] = self.cardinality.to_json()
        if self.required is not None:
            d["required"] = self.required
        if self.nullable is not None:
            d["nullable"] = self.nullable
        if self.default is not None:
            d["default"] = self.default
        if self.examples:
            d["examples"] = self.examples
        if self.constraints:
            d["constraints"] = [c.to_json() for c in self.constraints]
        if self.annotations:
            d["annotations"] = self.annotations
        if self.provenance:
            d["provenance"] = [p.to_json() for p in self.provenance]
        return d


@dataclass
class ChoiceIR:
    kind: str  # one_of/any_of/all_of
    options: List[Union[str, Dict[str, Any]]]
    cardinality: Optional[Cardinality] = None
    discriminator: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Provenance] = field(default_factory=list)

    def to_json(self) -> Json:
        d: Json = {"kind": self.kind, "options": self.options}
        if self.cardinality:
            d["cardinality"] = self.cardinality.to_json()
        if self.discriminator:
            d["discriminator"] = self.discriminator
        if self.annotations:
            d["annotations"] = self.annotations
        if self.provenance:
            d["provenance"] = [p.to_json() for p in self.provenance]
        return d


@dataclass
class EntityIR:
    id: str
    kind: str  # object/enum/string/number/integer/boolean/array/union
    name: Optional[str] = None
    description: Optional[str] = None
    abstract: Optional[bool] = None
    extends: List[Union[str, Dict[str, Any]]] = field(default_factory=list)
    fields: List[FieldIR] = field(default_factory=list)
    elementType: Optional[Union[str, Dict[str, Any]]] = None
    choice: Optional[ChoiceIR] = None
    constraints: List[Constraint] = field(default_factory=list)
    annotations: Dict[str, Any] = field(default_factory=dict)
    provenance: List[Provenance] = field(default_factory=list)

    def to_json(self) -> Json:
        d: Json = {"id": self.id, "kind": self.kind}
        if self.name:
            d["name"] = self.name
        if self.description:
            d["description"] = self.description
        if self.abstract is not None:
            d["abstract"] = self.abstract
        if self.extends:
            d["extends"] = self.extends
        if self.fields:
            d["fields"] = [f.to_json() for f in self.fields]
        if self.elementType is not None:
            d["elementType"] = self.elementType
        if self.choice is not None:
            d["choice"] = self.choice.to_json()
        if self.constraints:
            d["constraints"] = [c.to_json() for c in self.constraints]
        if self.annotations:
            d["annotations"] = self.annotations
        if self.provenance:
            d["provenance"] = [p.to_json() for p in self.provenance]
        return d


# -------------------------
# Utility helpers
# -------------------------

XSD_NS = "http://www.w3.org/2001/XMLSchema"
XS = f"{{{XSD_NS}}}"

XSD_PRIMITIVES = {
    "string": "string",
    "normalizedString": "string",
    "token": "string",
    "boolean": "boolean",
    "decimal": "number",
    "float": "number",
    "double": "number",
    "integer": "integer",
    "int": "integer",
    "long": "integer",
    "short": "integer",
    "nonNegativeInteger": "integer",
    "positiveInteger": "integer",
    "date": "string",
    "dateTime": "string",
    "time": "string",
    "anyURI": "string",
}


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://") or s.startswith("urn:")


def safe_id(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^A-Za-z0-9_.:#/-]+", "_", s)
    return s


def json_pointer_get(doc: Any, pointer: str) -> Any:
    if pointer.startswith("#"):
        pointer = pointer[1:]
    if pointer == "":
        return doc
    if not pointer.startswith("/"):
        raise ValueError(f"Unsupported JSON pointer: {pointer}")
    parts = pointer.split("/")[1:]
    cur = doc
    for p in parts:
        p = p.replace("~1", "/").replace("~0", "~")
        if isinstance(cur, dict):
            cur = cur[p]
        elif isinstance(cur, list):
            cur = cur[int(p)]
        else:
            raise KeyError(f"Pointer segment {p} not found")
    return cur


def match_patterns(path_rel: str, include_patterns: List[str], exclude_patterns: List[str]) -> bool:
    if include_patterns:
        ok = any(fnmatch.fnmatch(path_rel, pat) for pat in include_patterns)
        if not ok:
            return False
    if exclude_patterns:
        if any(fnmatch.fnmatch(path_rel, pat) for pat in exclude_patterns):
            return False
    return True


# -------------------------
# YAML bundle loading
# -------------------------

def read_bundle_manifest(bundle_dir: Path) -> Dict[str, Any]:
    yml = bundle_dir / "bundle.yaml"
    if not yml.exists():
        yml = bundle_dir / "bundle.yml"
    if not yml.exists():
        raise FileNotFoundError(f"Missing bundle.yaml in {bundle_dir}")
    return yaml.safe_load(yml.read_text(encoding="utf-8")) or {}


def bundle_name(manifest: Dict[str, Any]) -> str:
    name = manifest.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("bundle.yaml must contain a non-empty string field: 'name'")
    return name.strip()


def bundle_source_id(manifest: Dict[str, Any]) -> str:
    sid = manifest.get("source_id")
    return sid.strip() if isinstance(sid, str) and sid.strip() else bundle_name(manifest)


def bundle_version(manifest: Dict[str, Any]) -> Optional[str]:
    v = manifest.get("version")
    return v if isinstance(v, str) and v.strip() else None


# -------------------------
# JSON Schema handling
# -------------------------

class JsonSchemaResolver:
    def __init__(self, root_dir: Path, allow_external: bool, catalog: Dict[str, str]):
        self.root_dir = root_dir
        self.allow_external = allow_external
        self.catalog = catalog
        self.cache: Dict[Path, Any] = {}

    def load_doc(self, path: Path) -> Any:
        path = path.resolve()
        if path in self.cache:
            return self.cache[path]
        try:
            txt = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # fallback (rare, but happens)
            txt = path.read_text(encoding="utf-8", errors="replace")

        try:
            doc = json.loads(txt)
        except json.JSONDecodeError as e:
            # rethrow with path context
            raise json.JSONDecodeError(f"{e.msg} (file: {path})", e.doc, e.pos) from e

        self.cache[path] = doc
        return doc


    def resolve_ref(self, base_file: Path, ref: str) -> Tuple[Any, Path, str]:
        if is_url(ref):
            if ref in self.catalog:
                mapped = (self.root_dir / self.catalog[ref]).resolve()
                doc = self.load_doc(mapped)
                return doc, mapped, "#"
            if not self.allow_external:
                raise ValueError(f"External URL refs disallowed: {ref}")
            raise ValueError(f"URL refs not supported offline: {ref}")

        if ref.startswith("#"):
            doc = self.load_doc(base_file)
            target = json_pointer_get(doc, ref)
            return target, base_file, ref

        if "#" in ref:
            file_part, ptr = ref.split("#", 1)
            ptr = "#" + ptr
        else:
            file_part, ptr = ref, "#"

        target_file = (base_file.parent / file_part).resolve()
        try:
            doc = self.load_doc(target_file)
        except json.JSONDecodeError:
            if not self.allow_external:
                # treat as unresolved ref; caller can decide strictness
                return ({"$ref": ref, "x-unresolved": True}, target_file, ptr)
            raise
        target = json_pointer_get(doc, ptr)
        return target, target_file, ptr


class JsonSchemaToIR:
    def __init__(self, source_id: str, version: Optional[str], bundle_root: Path, allow_external: bool, catalog: Dict[str, str]):
        self.source_id = source_id
        self.version = version
        self.bundle_root = bundle_root
        self.resolver = JsonSchemaResolver(bundle_root, allow_external=allow_external, catalog=catalog)
        self.entities: Dict[str, EntityIR] = {}
        self._visiting: Set[str] = set()

    def build(self, entry_files: List[Path]) -> List[EntityIR]:
        for f in entry_files:
            self._schema_to_entities(f, self.resolver.load_doc(f), suggested_id=safe_id(f.stem))
        return list(self.entities.values())

    def _schema_to_entities(self, file_path: Path, schema: Any, suggested_id: str) -> str:

        if isinstance(schema, dict) and schema.get("x-unresolved") is True:
            return "any"

        if not isinstance(schema, dict):
            return "any"

        if "$ref" in schema:
            target, tfile, tptr = self.resolver.resolve_ref(file_path, schema["$ref"])
            ref_id = safe_id(f"{tfile.stem}{tptr}")
            return self._schema_to_entities(tfile, target, suggested_id=ref_id)

        schema_type = schema.get("type")
        title = schema.get("title")
        entity_id = safe_id(title) if isinstance(title, str) and title.strip() else safe_id(suggested_id)

        if entity_id in self._visiting:
            return entity_id
        if entity_id in self.entities:
            return entity_id

        if isinstance(schema_type, str) and schema_type in ("string", "number", "integer", "boolean") and not schema.get("properties") and not schema.get("enum"):
            if any(k in schema for k in ("pattern", "format", "minimum", "maximum", "minLength", "maxLength", "const")):
                pass
            else:
                return schema_type

        self._visiting.add(entity_id)
        ent = EntityIR(
            id=entity_id,
            kind=self._infer_kind(schema),
            name=title if isinstance(title, str) else None,
            description=schema.get("description"),
            annotations={"source_file": str(file_path.relative_to(self.bundle_root))},
            provenance=[Provenance(self.source_id, self.version, locator=str(file_path.relative_to(self.bundle_root)), note=_nowish_note())],
        )

        if "enum" in schema and isinstance(schema["enum"], list):
            ent.kind = "enum"
            ent.constraints.append(
                Constraint(kind="enum", values=schema["enum"], provenance=[Provenance(self.source_id, self.version, locator=str(file_path.relative_to(self.bundle_root)))])
            )

        if ent.kind == "object":
            props = schema.get("properties", {}) if isinstance(schema.get("properties"), dict) else {}
            required = set(schema.get("required", []) or [])
            for pname, psch in props.items():
                ent.fields.append(self._property_to_field(file_path, entity_id, pname, psch, pname in required))

        if ent.kind == "array":
            items = schema.get("items", {})
            elem_type = self._schema_to_entities(file_path, items, suggested_id=f"{entity_id}.items")
            ent.elementType = self._to_typeref(elem_type)

        union_keys = [k for k in ("oneOf", "anyOf", "allOf") if k in schema and isinstance(schema[k], list)]
        if union_keys:
            k = union_keys[0]
            options = []
            for idx, opt in enumerate(schema[k]):
                opt_type = self._schema_to_entities(file_path, opt, suggested_id=f"{entity_id}.{k}.{idx}")
                options.append(self._to_typeref(opt_type))
            ent.kind = "union"
            ent.choice = ChoiceIR(
                kind={"oneOf": "one_of", "anyOf": "any_of", "allOf": "all_of"}[k],
                options=options,
                provenance=[Provenance(self.source_id, self.version, locator=str(file_path.relative_to(self.bundle_root)))],
            )

        ent.constraints.extend(self._constraints_from_schema(file_path, entity_id, schema))

        self.entities[entity_id] = ent
        self._visiting.remove(entity_id)
        return entity_id

    def _property_to_field(self, file_path: Path, parent_id: str, pname: str, psch: Any, is_required: bool) -> FieldIR:
        if isinstance(psch, dict) and "$ref" in psch:
            target, tfile, tptr = self.resolver.resolve_ref(file_path, psch["$ref"])
            ptype_id = self._schema_to_entities(tfile, target, suggested_id=safe_id(f"{tfile.stem}{tptr}"))
            prov_locator = f"{file_path.relative_to(self.bundle_root)} :: $ref={psch['$ref']}"
        else:
            ptype_id = self._schema_to_entities(file_path, psch, suggested_id=f"{parent_id}.{pname}")
            prov_locator = f"{file_path.relative_to(self.bundle_root)} :: properties.{pname}"

        field_ir = FieldIR(
            name=pname,
            type=self._to_typeref(ptype_id),
            description=psch.get("description") if isinstance(psch, dict) else None,
            required=is_required,
            cardinality=Cardinality(min=1 if is_required else 0, max=1),
            provenance=[Provenance(self.source_id, self.version, locator=prov_locator)],
            annotations={},
        )

        if isinstance(psch, dict) and psch.get("type") == "array":
            field_ir.cardinality = Cardinality(min=1 if is_required else 0, max=None)

        if isinstance(psch, dict):
            field_ir.constraints.extend(self._constraints_from_schema(file_path, f"{parent_id}.{pname}", psch))
            if "default" in psch:
                field_ir.default = psch["default"]
            if "examples" in psch and isinstance(psch["examples"], list):
                field_ir.examples.extend(psch["examples"])
            if "const" in psch:
                field_ir.constraints.append(
                    Constraint(kind="const", const=psch["const"], target=f"{parent_id}.{pname}", provenance=[Provenance(self.source_id, self.version, locator=prov_locator)])
                )

        return field_ir

    def _constraints_from_schema(self, file_path: Path, target: str, schema: Dict[str, Any]) -> List[Constraint]:
        out: List[Constraint] = []
        prov = [Provenance(self.source_id, self.version, locator=str(file_path.relative_to(self.bundle_root)))]

        if "enum" in schema and isinstance(schema["enum"], list):
            out.append(Constraint(kind="enum", target=target, values=schema["enum"], provenance=prov))
        if "pattern" in schema and isinstance(schema["pattern"], str):
            out.append(Constraint(kind="pattern", target=target, pattern=schema["pattern"], provenance=prov))
        if "format" in schema and isinstance(schema["format"], str):
            out.append(Constraint(kind="format", target=target, format=schema["format"], provenance=prov))
        if "minimum" in schema or "maximum" in schema:
            out.append(Constraint(kind="range", target=target, minimum=schema.get("minimum"), maximum=schema.get("maximum"), provenance=prov))
        if "minLength" in schema or "maxLength" in schema:
            out.append(Constraint(kind="length", target=target, minLength=schema.get("minLength"), maxLength=schema.get("maxLength"), provenance=prov))
        return out

    def _infer_kind(self, schema: Dict[str, Any]) -> str:
        t = schema.get("type")
        if isinstance(t, list):
            non_null = [x for x in t if x != "null"]
            if len(non_null) == 1:
                return self._infer_kind({**schema, "type": non_null[0]})
            return "union"
        if t == "object" or "properties" in schema:
            return "object"
        if t == "array":
            return "array"
        if "oneOf" in schema or "anyOf" in schema or "allOf" in schema:
            return "union"
        if "enum" in schema:
            return "enum"
        if t in ("string", "number", "integer", "boolean"):
            return t
        return "object"

    def _to_typeref(self, type_id_or_prim: str) -> Union[str, Dict[str, Any]]:
        if type_id_or_prim in ("string", "number", "integer", "boolean", "any"):
            return type_id_or_prim
        return {"ref": type_id_or_prim}


# -------------------------
# XSD handling
# -------------------------

class XsdBundleResolver:
    def __init__(self, root_dir: Path, allow_external: bool, catalog: Dict[str, str]):
        self.root_dir = root_dir
        self.allow_external = allow_external
        self.catalog = catalog
        self.docs: Dict[Path, etree._ElementTree] = {}

    def load_tree(self, path: Path) -> etree._ElementTree:
        path = path.resolve()
        if path in self.docs:
            return self.docs[path]
        parser = etree.XMLParser(remove_comments=False, resolve_entities=False, huge_tree=True)
        tree = etree.parse(str(path), parser)
        self.docs[path] = tree
        return tree

    def map_location(self, base_file: Path, schema_location: str) -> Optional[Path]:
        if is_url(schema_location):
            if schema_location in self.catalog:
                return (self.root_dir / self.catalog[schema_location]).resolve()
            if not self.allow_external:
                return None
            # offline: no fetch
            return None
        return (base_file.parent / schema_location).resolve()

    def resolve_all(self, entry_files: List[Path], follow_includes: bool, follow_imports: bool) -> List[Path]:
        loaded: Set[Path] = set()
        stack: List[Path] = [p.resolve() for p in entry_files]

        while stack:
            p = stack.pop()
            if p in loaded:
                continue
            tree = self.load_tree(p)
            loaded.add(p)

            root = tree.getroot()
            if follow_includes:
                for node in root.findall(f"{XS}include"):
                    loc = node.get("schemaLocation")
                    if not loc:
                        continue
                    candidate = self.map_location(p, loc)
                    if candidate and candidate.exists():
                        stack.append(candidate)

            if follow_imports:
                for node in root.findall(f"{XS}import"):
                    loc = node.get("schemaLocation")
                    if not loc:
                        continue
                    candidate = self.map_location(p, loc)
                    if candidate and candidate.exists():
                        stack.append(candidate)

        return sorted(list(loaded))


class XsdToIR:
    def __init__(self, source_id: str, version: Optional[str], bundle_root: Path, allow_external: bool, catalog: Dict[str, str], follow_includes: bool, follow_imports: bool):
        self.source_id = source_id
        self.version = version
        self.bundle_root = bundle_root
        self.resolver = XsdBundleResolver(bundle_root, allow_external=allow_external, catalog=catalog)
        self.follow_includes = follow_includes
        self.follow_imports = follow_imports

        self.complex_types: Dict[Tuple[str, str], etree._Element] = {}
        self.simple_types: Dict[Tuple[str, str], etree._Element] = {}
        self.global_elements: Dict[Tuple[str, str], etree._Element] = {}

        self.entities: Dict[str, EntityIR] = {}

    def build(self, entry_files: List[Path]) -> List[EntityIR]:
        all_files = self.resolver.resolve_all(entry_files, self.follow_includes, self.follow_imports)

        for f in all_files:
            tree = self.resolver.load_tree(f)
            root = tree.getroot()
            tns = root.get("targetNamespace", "") or ""
            for ct in root.findall(f"{XS}complexType"):
                name = ct.get("name")
                if name:
                    self.complex_types[(tns, name)] = ct
            for st in root.findall(f"{XS}simpleType"):
                name = st.get("name")
                if name:
                    self.simple_types[(tns, name)] = st
            for el in root.findall(f"{XS}element"):
                name = el.get("name")
                if name:
                    self.global_elements[(tns, name)] = el

        for (tns, name), node in self.simple_types.items():
            self._emit_simple_type(tns, name, node)

        for (tns, name), node in self.complex_types.items():
            self._emit_complex_type(tns, name, node)

        for (tns, name), node in self.global_elements.items():
            self._emit_global_element_as_entity(tns, name, node)

        return list(self.entities.values())

    def _qid(self, tns: str, name: str) -> str:
        if tns:
            return safe_id(f"{self.source_id}:{tns}#{name}")
        return safe_id(f"{self.source_id}:{name}")

    def _prov(self, locator: str) -> List[Provenance]:
        return [Provenance(self.source_id, self.version, locator=locator, note=_nowish_note())]

    def _emit_simple_type(self, tns: str, name: str, node: etree._Element) -> str:
        eid = self._qid(tns, name)
        if eid in self.entities:
            return eid

        restriction = node.find(f"{XS}restriction")
        enum_vals: List[str] = []
        pattern = None
        base = restriction.get("base") if restriction is not None else None
        if restriction is not None:
            for e in restriction.findall(f"{XS}enumeration"):
                v = e.get("value")
                if v is not None:
                    enum_vals.append(v)
            pat = restriction.find(f"{XS}pattern")
            if pat is not None:
                pattern = pat.get("value")

        kind = "enum" if enum_vals else "string"
        ent = EntityIR(
            id=eid,
            kind=kind,
            description=None,
            annotations={"xsd_targetNamespace": tns, "xsd_name": name},
            provenance=self._prov(f"xsd:simpleType:{name}"),
        )

        if enum_vals:
            ent.constraints.append(Constraint(kind="enum", values=enum_vals, provenance=self._prov(f"xsd:simpleType:{name}")))
        if pattern:
            ent.constraints.append(Constraint(kind="pattern", pattern=pattern, provenance=self._prov(f"xsd:simpleType:{name}")))
        if base:
            ent.annotations["xsd_base"] = base

        self.entities[eid] = ent
        return eid

    def _emit_complex_type(self, tns: str, name: str, node: etree._Element) -> str:
        eid = self._qid(tns, name)
        if eid in self.entities:
            return eid

        ent = EntityIR(
            id=eid,
            kind="object",
            name=name,
            description=None,
            annotations={"xsd_targetNamespace": tns, "xsd_name": name},
            provenance=self._prov(f"xsd:complexType:{name}"),
        )

        ext = node.find(f".//{XS}extension")
        if ext is not None and ext.get("base"):
            base = ext.get("base")
            base_ref = self._resolve_qname_to_entity_ref(node, base)
            ent.extends.append({"ref": base_ref} if base_ref else base)

        # handle sequence/all
        
        content = node.find(f"{XS}sequence")
        if content is None:
            content = node.find(f"{XS}all")
        if content is None:
            content = node.find(f"{XS}choice")

        if content is not None:
            for child in content.findall(f"{XS}element"):
                f_ir = self._xsd_element_to_field(tns, child, parent_entity_id=eid)
                if f_ir:
                    ent.fields.append(f_ir)

        self.entities[eid] = ent
        return eid

    def _emit_global_element_as_entity(self, tns: str, name: str, node: etree._Element) -> Optional[str]:
        inline_ct = node.find(f"{XS}complexType")
        inline_st = node.find(f"{XS}simpleType")
        if inline_ct is None and inline_st is None:
            return None

        eid = self._qid(tns, name)
        if eid in self.entities:
            return eid

        if inline_st is not None:
            self.simple_types[(tns, name)] = inline_st
            return self._emit_simple_type(tns, name, inline_st)

        ent = EntityIR(
            id=eid,
            kind="object",
            name=name,
            description=None,
            annotations={"xsd_targetNamespace": tns, "xsd_globalElement": name},
            provenance=self._prov(f"xsd:element:{name}"),
        )
        content = inline_ct.find(f"{XS}sequence") or inline_ct.find(f"{XS}all") or inline_ct.find(f"{XS}choice")
        if content is not None:
            for child in content.findall(f"{XS}element"):
                f_ir = self._xsd_element_to_field(tns, child, parent_entity_id=eid)
                if f_ir:
                    ent.fields.append(f_ir)
        self.entities[eid] = ent
        return eid

    def _xsd_element_to_field(self, tns: str, el: etree._Element, parent_entity_id: str) -> Optional[FieldIR]:
        name = el.get("name")
        ref = el.get("ref")
        if not name and not ref:
            return None

        mino = int(el.get("minOccurs", "1"))
        maxo_raw = el.get("maxOccurs", "1")
        maxo = None if maxo_raw == "unbounded" else int(maxo_raw)

        type_attr = el.get("type")
        inline_ct = el.find(f"{XS}complexType")
        inline_st = el.find(f"{XS}simpleType")

        field_name = name if name else ref.split(":")[-1]
        locator = f"xsd:field:{parent_entity_id}.{field_name}"

        if ref and not type_attr and inline_ct is None and inline_st is None:
            qname = self._resolve_qname(el, ref)
            if qname:
                rt_ns, rt_local = qname
                g = self.global_elements.get((rt_ns, rt_local))
                if g is not None:
                    type_attr = g.get("type")
                    inline_ct = g.find(f"{XS}complexType")
                    inline_st = g.find(f"{XS}simpleType")

        if inline_st is not None:
            st_eid = self._emit_simple_type(tns, f"{parent_entity_id}.{field_name}", inline_st)
            ftype: Union[str, Dict[str, Any]] = {"ref": st_eid}
        elif inline_ct is not None:
            ct_name = safe_id(f"{parent_entity_id}.{field_name}")
            ct_eid = self._qid(tns, ct_name)
            if ct_eid not in self.entities:
                ent = EntityIR(
                    id=ct_eid,
                    kind="object",
                    name=ct_name,
                    annotations={"xsd_inline_of": parent_entity_id},
                    provenance=self._prov(f"xsd:inlineComplexType:{parent_entity_id}.{field_name}"),
                )
                content = inline_ct.find(f"{XS}sequence") or inline_ct.find(f"{XS}all") or inline_ct.find(f"{XS}choice")
                if content is not None:
                    for child in content.findall(f"{XS}element"):
                        cf = self._xsd_element_to_field(tns, child, parent_entity_id=ct_eid)
                        if cf:
                            ent.fields.append(cf)
                self.entities[ct_eid] = ent
            ftype = {"ref": ct_eid}
        else:
            if type_attr:
                resolved = self._resolve_qname_to_entity_ref(el, type_attr)
                if resolved:
                    ftype = resolved if resolved in ("string", "number", "integer", "boolean") else {"ref": resolved}
                else:
                    prim = self._xsd_primitive_from_qname(type_attr)
                    ftype = prim if prim else "string"
            else:
                ftype = "string"

        required = (mino >= 1)
        return FieldIR(
            name=field_name,
            type=ftype,
            required=required,
            cardinality=Cardinality(min=mino, max=maxo),
            provenance=self._prov(locator),
            annotations={},
        )

    def _xsd_primitive_from_qname(self, qname_str: str) -> Optional[str]:
        local = qname_str.split(":")[-1]
        if local in XSD_PRIMITIVES:
            return XSD_PRIMITIVES[local]
        return None

    def _resolve_qname(self, context_el: etree._Element, qname_str: str) -> Optional[Tuple[str, str]]:
        if qname_str.startswith("{"):
            m = re.match(r"^\{([^}]+)\}(.+)$", qname_str)
            if not m:
                return None
            return m.group(1), m.group(2)

        if ":" in qname_str:
            prefix, local = qname_str.split(":", 1)
            ns = context_el.nsmap.get(prefix)
            if ns is None:
                return None
            return ns, local

        return "", qname_str

    def _resolve_qname_to_entity_ref(self, context_el: etree._Element, qname_str: str) -> Optional[str]:
        qn = self._resolve_qname(context_el, qname_str)
        if qn is None:
            return None
        ns, local = qn

        if ns == XSD_NS and local in XSD_PRIMITIVES:
            return XSD_PRIMITIVES[local]
        if (ns, local) in self.complex_types:
            return self._qid(ns, local)
        if (ns, local) in self.simple_types:
            return self._qid(ns, local)
        if (ns, local) in self.global_elements:
            return self._qid(ns, local)
        return None


# -------------------------
# Examples enrichment
# -------------------------

def load_examples(examples_dir: Path, xml_pats: List[str], json_pats: List[str]) -> List[Tuple[str, Any, str]]:
    out: List[Tuple[str, Any, str]] = []
    if not examples_dir.exists():
        return out

    for p in sorted(examples_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(examples_dir)).replace("\\", "/")
        if p.suffix.lower() == ".json" and any(fnmatch.fnmatch(rel, pat) for pat in (json_pats or ["**/*.json"])):
            try:
                out.append(("json", json.loads(p.read_text(encoding="utf-8")), p.name))
            except Exception:
                out.append(("json", p.read_text(encoding="utf-8"), p.name))
        elif p.suffix.lower() == ".xml" and any(fnmatch.fnmatch(rel, pat) for pat in (xml_pats or ["**/*.xml"])):
            try:
                parser = etree.XMLParser(remove_comments=False, resolve_entities=False, huge_tree=True)
                root = etree.fromstring(p.read_bytes(), parser)
                out.append(("xml", root, p.name))
            except Exception:
                out.append(("xml", p.read_text(encoding="utf-8"), p.name))
    return out


def enrich_entities_with_examples(entities: Dict[str, EntityIR], examples: List[Tuple[str, Any, str]]) -> None:
    def suffixes(eid: str) -> Set[str]:
        parts = re.split(r"[#.:/]+", eid)
        return {p for p in parts if p}

    name_map: Dict[str, List[str]] = {}
    for eid in entities:
        for s in suffixes(eid):
            name_map.setdefault(s.lower(), []).append(eid)

    for kind, obj, fname in examples:
        if kind == "json" and isinstance(obj, dict) and len(obj) == 1:
            k = next(iter(obj.keys()))
            for eid in name_map.get(str(k).lower(), []):
                entities[eid].annotations.setdefault("examples_index", []).append({"file": fname, "kind": "json", "hint": k})
        elif kind == "xml" and isinstance(obj, etree._Element):
            local = etree.QName(obj).localname
            for eid in name_map.get(local.lower(), []):
                entities[eid].annotations.setdefault("examples_index", []).append({"file": fname, "kind": "xml", "hint": local})


# -------------------------
# Deep inlining (cycle-safe) driven by bundle.yaml
# -------------------------

def _is_typeref_ref(x: Any) -> Optional[str]:
    if isinstance(x, dict) and "ref" in x and isinstance(x["ref"], str):
        return x["ref"]
    return None


def inline_entities(entities: Dict[str, EntityIR], roots: Optional[List[str]], max_depth: Optional[int]) -> None:
    cache: Dict[Tuple[str, int], Any] = {}
    root_set = set(roots) if roots else None

    def inline_entity(eid: str, stack: List[str], depth: int) -> Any:
        if max_depth is not None and depth > max_depth:
            return {"ref": eid, "annotations": {"depth_cutoff": True}}
        key = (eid, depth)
        if key in cache:
            return cache[key]
        if eid in stack:
            return {"ref": eid, "annotations": {"cycle": True}}
        ent = entities.get(eid)
        if ent is None:
            return {"ref": eid, "annotations": {"missing": True}}

        stack2 = stack + [eid]
        out: Dict[str, Any] = {"id": ent.id, "kind": ent.kind}
        if ent.name:
            out["name"] = ent.name
        if ent.description:
            out["description"] = ent.description
        if ent.abstract is not None:
            out["abstract"] = ent.abstract

        if ent.extends:
            out["extends"] = []
            for tr in ent.extends:
                rid = _is_typeref_ref(tr)
                out["extends"].append(inline_entity(rid, stack2, depth + 1) if rid else tr)

        if ent.kind == "object":
            out_fields = []
            for f in ent.fields:
                fin: Dict[str, Any] = {"name": f.name}
                if f.description:
                    fin["description"] = f.description
                if f.required is not None:
                    fin["required"] = f.required
                if f.nullable is not None:
                    fin["nullable"] = f.nullable
                if f.cardinality is not None:
                    fin["cardinality"] = f.cardinality.to_json()
                if f.default is not None:
                    fin["default"] = f.default
                if f.examples:
                    fin["examples"] = f.examples
                if f.constraints:
                    fin["constraints"] = [c.to_json() for c in f.constraints]

                rid = _is_typeref_ref(f.type)
                fin["type"] = inline_entity(rid, stack2, depth + 1) if rid else f.type
                out_fields.append(fin)
            out["fields"] = out_fields

        elif ent.kind == "array":
            if ent.elementType is not None:
                rid = _is_typeref_ref(ent.elementType)
                out["elementType"] = inline_entity(rid, stack2, depth + 1) if rid else ent.elementType

        elif ent.kind == "union" and ent.choice is not None:
            out_choice: Dict[str, Any] = {"kind": ent.choice.kind, "options": []}
            if ent.choice.cardinality is not None:
                out_choice["cardinality"] = ent.choice.cardinality.to_json()
            if ent.choice.discriminator:
                out_choice["discriminator"] = ent.choice.discriminator
            for opt in ent.choice.options:
                rid = _is_typeref_ref(opt)
                out_choice["options"].append(inline_entity(rid, stack2, depth + 1) if rid else opt)
            out["choice"] = out_choice

        if ent.constraints:
            out["constraints"] = [c.to_json() for c in ent.constraints]

        cache[key] = out
        return out

    targets = list(entities.keys()) if root_set is None else [eid for eid in entities.keys() if eid in root_set]

    for eid in targets:
        entities[eid].annotations["inline"] = inline_entity(eid, [], depth=0)
        entities[eid].annotations["inline_config"] = {"max_depth": max_depth, "cycle_behavior": "emit_ref", "mode": "roots" if root_set else "all"}


# -------------------------
# Cross-bundle mappings
# -------------------------

def load_ir_file(bundle_dir: Path, bundle_name: str) -> Optional[Dict[str, Any]]:
    p = bundle_dir / f"{bundle_name}_ir.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def index_ir_entities(ir: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for e in ir.get("entities", []) or []:
        if isinstance(e, dict) and isinstance(e.get("id"), str):
            idx[e["id"]] = e
    return idx


def apply_mappings_to_ir(
    this_ir: Dict[str, Any],
    this_bundle_name: str,
    mapping_rules: Dict[str, Any],
    other_bundle_irs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Returns mapping artifact JSON and also annotates entities inside this_ir.
    """
    mappings = mapping_rules.get("mappings", []) if isinstance(mapping_rules, dict) else []
    if not isinstance(mappings, list):
        mappings = []

    # Build indices
    this_idx = index_ir_entities(this_ir)
    other_idx: Dict[str, Dict[str, Dict[str, Any]]] = {bn: index_ir_entities(ir) for bn, ir in other_bundle_irs.items()}

    artifact: Dict[str, Any] = {
        "bundle": this_bundle_name,
        "mappings": [],
    }

    for m in mappings:
        if not isinstance(m, dict):
            continue
        kind = m.get("kind")
        frm = m.get("from", {})
        to = m.get("to", {})
        if not isinstance(frm, dict) or not isinstance(to, dict):
            continue

        frm_bundle = frm.get("bundle")
        to_bundle = to.get("bundle")
        if frm_bundle != this_bundle_name:
            # for now, we only “apply” mappings originating from this bundle
            continue

        rec = {
            "kind": kind,
            "from": frm,
            "to": to,
            "relation": m.get("relation"),
            "confidence": m.get("confidence"),
            "note": m.get("note"),
        }
        artifact["mappings"].append(rec)

        # Annotate entity in this bundle
        frm_eid = frm.get("entity")
        if isinstance(frm_eid, str) and frm_eid in this_idx:
            this_idx[frm_eid].setdefault("annotations", {}).setdefault("cross_bundle_mappings", []).append(rec)

        # Optionally validate target exists
        if isinstance(to_bundle, str) and isinstance(to.get("entity"), str):
            eid2 = to["entity"]
            if to_bundle in other_idx and eid2 not in other_idx[to_bundle]:
                rec.setdefault("warnings", []).append(f"Target entity not found in bundle {to_bundle}: {eid2}")

    return artifact


# -------------------------
# Discovery driven by bundle.yaml
# -------------------------

def discover_files(root_dir: Path, include_patterns: List[str], exclude_patterns: List[str], suffixes: Set[str]) -> List[Path]:
    """
    Robust discovery:
    - uses Path.glob so ** patterns work exactly like shell globs
    - supports exclude_patterns via fnmatch on relative paths
    """
    out: Set[Path] = set()
    if not root_dir.exists():
        return []

    include_patterns = include_patterns or ["**/*"]

    # include
    for pat in include_patterns:
        for p in root_dir.glob(pat):
            if p.is_file() and p.suffix.lower() in suffixes:
                out.add(p)

    # exclude
    if exclude_patterns:
        filtered: Set[Path] = set()
        for p in out:
            rel = str(p.relative_to(root_dir)).replace("\\", "/")
            if any(fnmatch.fnmatch(rel, ep) for ep in exclude_patterns):
                continue
            filtered.add(p)
        out = filtered

    return sorted(out)



# -------------------------
# Orchestration
# -------------------------

def build_ir(bundle_dir: Path, out_dir: Optional[Path] = None) -> Tuple[Path, Dict[str, Any]]:
    bundle_dir = bundle_dir.resolve()
    manifest = read_bundle_manifest(bundle_dir)

    name = bundle_name(manifest)
    source_id = bundle_source_id(manifest)
    version = bundle_version(manifest)

    strict = bool((manifest.get("build") or {}).get("strict", True))
    ir_version = str((manifest.get("build") or {}).get("ir_version", "0.2"))

    # catalog mapping
    catalog = (manifest.get("resolution") or {}).get("catalog") or {}
    if not isinstance(catalog, dict):
        catalog = {}

    # schema discovery config
    schemas_cfg = manifest.get("schemas") or {}
    xsd_cfg = (schemas_cfg.get("xsd") or {}) if isinstance(schemas_cfg, dict) else {}
    js_cfg = (schemas_cfg.get("jsonschema") or {}) if isinstance(schemas_cfg, dict) else {}

    xsd_root = bundle_dir / (xsd_cfg.get("root_dir") or "schemas")
    js_root = bundle_dir / (js_cfg.get("root_dir") or "schemas")

    xsd_include = xsd_cfg.get("include_patterns") or ["**/*.xsd"]
    xsd_exclude = xsd_cfg.get("exclude_patterns") or []
    js_include = js_cfg.get("include_patterns") or ["**/*.json"]
    js_exclude = js_cfg.get("exclude_patterns") or []

    xsd_files = discover_files(xsd_root, xsd_include, xsd_exclude, {".xsd"})
    js_files = discover_files(js_root, js_include, js_exclude, {".json"})

    if strict:
        print(f"[bundle] {name} at {bundle_dir}")
        print(f"[discover] xsd_root={xsd_root} -> {len(xsd_files)} files")
        print(f"[discover] js_root={js_root} -> {len(js_files)} files")
        if len(xsd_files) < 5:
            for p in xsd_files:
                print("  xsd:", p)
        if len(js_files) < 5:
            for p in js_files:
                print("  json:", p)


    # entrypoints
    entry_xsds: List[Path] = []
    entry_jsons: List[Path] = []
    if isinstance(manifest.get("entrypoints"), list) and manifest["entrypoints"]:
        for ep in manifest["entrypoints"]:
            if not isinstance(ep, dict):
                continue
            kind = ep.get("kind")
            rel = ep.get("path")
            if not isinstance(rel, str):
                continue
            p = (bundle_dir / rel).resolve()
            if kind == "xsd":
                entry_xsds.append(p)
            elif kind in ("jsonschema", "json_schema", "json"):
                entry_jsons.append(p)
    else:
        entry_xsds = xsd_files[:]
        entry_jsons = js_files[:]

    # resolution options
    res = manifest.get("resolution") or {}
    xsd_res = res.get("xsd") or {}
    js_res = res.get("jsonschema") or {}
    follow_includes = bool(xsd_res.get("follow_includes", True))
    follow_imports = bool(xsd_res.get("follow_imports", True))
    xsd_allow_external = bool(xsd_res.get("allow_external", False))
    js_allow_external = bool(js_res.get("allow_external", False))

    entities: Dict[str, EntityIR] = {}

    # Build XSD entities
    if entry_xsds:
        xsd_builder = XsdToIR(
            source_id=source_id,
            version=version,
            bundle_root=bundle_dir,
            allow_external=xsd_allow_external,
            catalog=catalog,
            follow_includes=follow_includes,
            follow_imports=follow_imports,
        )
        for e in xsd_builder.build(entry_xsds):
            entities[e.id] = e

    # Build JSON Schema entities
    if entry_jsons:
        js_builder = JsonSchemaToIR(
            source_id=source_id,
            version=version,
            bundle_root=bundle_dir,
            allow_external=js_allow_external,
            catalog=catalog,
        )

        bad_files: List[str] = []
        for f in entry_jsons:
            try:
                js_builder._schema_to_entities(f, js_builder.resolver.load_doc(f), suggested_id=safe_id(f.stem))
            except json.JSONDecodeError as e:
                bad_files.append(str(f))
                if strict:
                    raise
            except Exception:
                bad_files.append(str(f))
                if strict:
                    raise

        # merge entities
        for e in js_builder.entities.values():
            if e.id in entities:
                e.id = safe_id(f"{source_id}:{e.id}")
            entities[e.id] = e

        if bad_files:
            ir_warnings = (manifest.get("build") or {}).setdefault("warnings", [])
            # also store on IR bundle annotations later; but at least print:
            print(f"[warn] skipped {len(bad_files)} invalid/failed json files (strict={strict})")
            for p in bad_files[:10]:
                print("  -", p)

    # Examples
    ex_cfg = manifest.get("examples") or {}
    ex_root = bundle_dir / (ex_cfg.get("root_dir") or "examples")
    xml_pats = ((ex_cfg.get("xml") or {}).get("include_patterns") or ["**/*.xml"]) if isinstance(ex_cfg, dict) else ["**/*.xml"]
    json_pats = ((ex_cfg.get("json") or {}).get("include_patterns") or ["**/*.json"]) if isinstance(ex_cfg, dict) else ["**/*.json"]
    examples = load_examples(ex_root, xml_pats=xml_pats, json_pats=json_pats)
    if examples:
        enrich_entities_with_examples(entities, examples)

    # Inline (driven by bundle.yaml)
    inline_cfg = ((manifest.get("normalization") or {}).get("inline") or {})
    if isinstance(inline_cfg, dict) and bool(inline_cfg.get("enabled", False)):
        roots = inline_cfg.get("roots") or []
        if not isinstance(roots, list):
            roots = []
        roots = [r for r in roots if isinstance(r, str) and r.strip()]
        max_depth = inline_cfg.get("max_depth")
        if isinstance(max_depth, int):
            md = max_depth
        else:
            md = None
        inline_entities(entities, roots=roots if roots else None, max_depth=md)

    # Emit IR file
    out_dir = out_dir.resolve() if out_dir else bundle_dir
    ir_path = out_dir / f"{name}_ir.json"

    ir: Dict[str, Any] = {
        "ir_version": ir_version,
        "source_bundle": {
            "name": name,
            "source_id": source_id,
            "version": version,
            "bundle_dir": str(bundle_dir),
            "generator": (manifest.get("build") or {}).get("generator", "rag-utils-ir-builder"),
            "notes": _nowish_note(),
            "counts": {
                "entities": len(entities),
                "xsd_files_discovered": len(xsd_files),
                "json_schema_files_discovered": len(js_files),
                "examples_discovered": len(examples),
            },
            "annotations": manifest.get("annotations") or {},
        },
        "entities": [entities[k].to_json() for k in sorted(entities.keys())],
    }

    ir_path.write_text(json.dumps(ir, indent=2, ensure_ascii=False), encoding="utf-8")

    # -------------------------
    # Cross-bundle mappings
    # -------------------------
    mappings_cfg = manifest.get("mappings") or {}
    if isinstance(mappings_cfg, dict) and bool(mappings_cfg.get("enabled", False)):
        rules_rel = mappings_cfg.get("rules") or "mappings.yaml"
        rules_path = (bundle_dir / rules_rel).resolve()
        if not rules_path.exists():
            if strict:
                raise FileNotFoundError(f"Mappings enabled but rules file not found: {rules_path}")
        else:
            rules = yaml.safe_load(rules_path.read_text(encoding="utf-8")) or {}

            # Load or build other bundle IRs
            other_irs: Dict[str, Dict[str, Any]] = {}
            bundles = mappings_cfg.get("bundles") or []
            build_missing = bool(mappings_cfg.get("build_missing_bundles", True))

            if isinstance(bundles, list):
                for b in bundles:
                    if not isinstance(b, dict):
                        continue
                    other_path = b.get("path")
                    other_name = b.get("name")
                    if not isinstance(other_path, str) or not isinstance(other_name, str):
                        continue
                    odir = (bundle_dir / other_path).resolve()
                    ir2 = load_ir_file(odir, other_name)
                    if ir2 is None and build_missing:
                        # build the other bundle first
                        _, _ = build_ir(odir, out_dir=None)
                        ir2 = load_ir_file(odir, other_name)
                    if ir2 is None:
                        if strict:
                            raise FileNotFoundError(f"Could not load IR for mapped bundle {other_name} at {odir}")
                        continue
                    other_irs[other_name] = ir2

            # Apply mappings (also annotates this_ir)
            mapping_artifact = apply_mappings_to_ir(ir, name, rules, other_irs)

            # Write updated IR (with annotations applied)
            ir_path.write_text(json.dumps(ir, indent=2, ensure_ascii=False), encoding="utf-8")

            # Write mapping artifact
            map_path = out_dir / f"{name}_mappings.json"
            map_path.write_text(json.dumps(mapping_artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    return ir_path, ir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle_dir", type=str, help="Path to bundle folder containing bundle.yaml")
    ap.add_argument("--out", type=str, default=None, help="Optional output directory")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else None
    ir_path, _ = build_ir(Path(args.bundle_dir), out_dir=out_dir)
    print(f"[generated IR] {ir_path}")
    


if __name__ == "__main__":
    main()
