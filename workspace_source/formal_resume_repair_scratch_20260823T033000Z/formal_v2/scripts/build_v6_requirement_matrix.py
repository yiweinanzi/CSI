from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import re
import tempfile

from formal_v2.scripts.v6_trace_registry import (
    TRACE_FAMILIES,
    evidence_family_for_clause,
)


SOURCE_SPECS = {
    "reader": {
        "sha256": "5866888fac736bcb812ebe3630b38095ad4989979a9fdf68cabcdbfe286f737e",
        "prefix": "ZR",
    },
}

NORMATIVE_TERMS = re.compile(
    "|".join(
        (
            "必须", "只能", "不得", "禁止", "至少", "唯一", "冻结", "等权",
            "非劣", "等效", "主指标", "资格门", "通过", "失败", "停止", "信息预算",
            "不能", "不允许", "应当", "需要", "固定", "共享", "排除", "不参与",
            "只用", "定义", "默认", "阈值", "条件", "分别", "保持", "严格",
        )
    )
)
FORMULA_MARKERS = re.compile(
    r"(?:\\begin\{|\\operatorname|\\mathbb|\\Delta|\\epsilon|\\tau|<=|>=|≤|≥|:=|=)"
)
TOP_SECTION = re.compile(r"^##\s+(\d+)(?:\.|\s)")
HEADING = re.compile(r"^#{2,6}\s+")
SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？；;])\s*")
TRACKED_SECTIONS = frozenset(range(16))

FIELDNAMES = (
    "requirement_id",
    "source_document",
    "source_document_sha256",
    "source_line",
    "clause_index",
    "source_clause_sha256",
    "original_norm",
    "normative_signal",
    "evidence_family",
    "mapping_basis",
    "section",
    "section_heading",
    "verifiable_expectation",
    "runtime_entry",
    "code_locations",
    "config_or_schema_locations",
    "regression_test_locations",
    "dynamic_evidence",
    "paper_location",
    "status",
    "blocking_type",
    "repair_or_boundary",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate(project_root: Path, items: tuple[tuple[str, str], ...]) -> str:
    locations = []
    for relative, anchor in items:
        path = project_root / relative
        lines = path.read_text(encoding="utf-8").splitlines()
        matches = [index for index, line in enumerate(lines, start=1) if anchor in line]
        if not matches:
            raise RuntimeError(f"traceability anchor not found: {relative}:{anchor}")
        locations.append(f"{relative}:{matches[0]}")
    return "; ".join(locations)


def clauses_for_line(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped or stripped in {"---", "```", "\\[", "\\]"}:
        return []
    if stripped.startswith("```"):
        return []
    if stripped.startswith("|"):
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        return [
            cell
            for cell in cells
            if cell and re.fullmatch(r"[-: ]+", cell) is None
        ]
    stripped = HEADING.sub("", stripped)
    stripped = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)", "", stripped)
    return [part for part in SENTENCE_BOUNDARY.split(stripped) if part.strip()]


def normative_signal(
    clause: str,
    raw_line: str,
    *,
    table_requirement: bool = False,
) -> str:
    terms = sorted(set(NORMATIVE_TERMS.findall(clause)))
    if terms:
        return "TERM:" + ",".join(terms)
    if FORMULA_MARKERS.search(clause):
        return "FORMULA_OR_CONDITION"
    if table_requirement:
        return "NORMATIVE_TABLE_CELL"
    if HEADING.match(raw_line.strip()):
        return "HEADING_INCLUDED_FOR_COMPLETENESS"
    return "CONTEXT_INCLUDED_FOR_COMPLETENESS"


def is_normative_signal(signal: str) -> bool:
    return (
        signal.startswith("TERM:")
        or signal in {"FORMULA_OR_CONDITION", "NORMATIVE_TABLE_CELL"}
    )


def _is_table_header(lines: list[str], index: int) -> bool:
    if not lines[index].strip().startswith("|") or index + 1 >= len(lines):
        return False
    next_line = lines[index + 1].strip()
    if not next_line.startswith("|"):
        return False
    cells = [cell.strip() for cell in next_line.strip("|").split("|")]
    return bool(cells) and all(
        cell and re.fullmatch(r"[-: ]+", cell) is not None for cell in cells
    )


def _context_mapping(clause_digest: str) -> dict[str, str]:
    return {
        "evidence_family": "not_normative_context",
        "mapping_basis": f"context-classification;clause-sha256:{clause_digest}",
        "verifiable_expectation": (
            "Context is retained for source-line completeness and is not promoted as "
            "an atomic normative requirement."
        ),
        "runtime_entry": "NOT_APPLICABLE_CONTEXT",
        "code_locations": "NOT_APPLICABLE_CONTEXT",
        "config_or_schema_locations": "NOT_APPLICABLE_CONTEXT",
        "regression_test_locations": "NOT_APPLICABLE_CONTEXT",
        "dynamic_evidence": "NOT_APPLICABLE_CONTEXT",
        "paper_location": "NOT_APPLICABLE_CONTEXT",
        "status": "PARTIAL/PROXY",
        "blocking_type": "NOT_NORMATIVE_CONTEXT",
        "repair_or_boundary": (
            "No code implementation is required unless the author designates this "
            "context clause as normative."
        ),
    }


def _validate_family(name: str) -> None:
    family = TRACE_FAMILIES[name]
    if family.status not in {"EXACT", "PARTIAL/PROXY", "MISSING", "CONFLICT"}:
        raise RuntimeError(f"semantic trace family {name!r} has invalid status")
    if family.status == "EXACT" and not all(
        (family.entry, family.code, family.config, family.tests, family.dynamic)
    ):
        raise RuntimeError(
            f"EXACT semantic trace family {name!r} lacks executable evidence"
        )


def _normative_mapping(
    source_name: str,
    line_number: int,
    section: int,
    heading: str,
    clause: str,
    clause_digest: str,
    project_root: Path,
) -> dict[str, str]:
    family_name = (
        "results"
        if section == 13
        else evidence_family_for_clause(section, heading, clause)
    )
    if not family_name or family_name not in TRACE_FAMILIES:
        raise RuntimeError(
            f"normative clause has no semantic trace family: "
            f"{source_name}:{line_number}:{clause_digest}"
        )
    _validate_family(family_name)
    family = TRACE_FAMILIES[family_name]
    mapping = {
        "evidence_family": family_name,
        "mapping_basis": (
            f"semantic-family:{family_name};clause-sha256:{clause_digest}"
        ),
        "verifiable_expectation": family.expectation,
        "runtime_entry": family.entry,
        "code_locations": locate(project_root, family.code),
        "config_or_schema_locations": locate(project_root, family.config),
        "regression_test_locations": locate(project_root, family.tests),
        "dynamic_evidence": family.dynamic,
        "paper_location": family.paper,
        "status": family.status,
        "blocking_type": family.blocker,
        "repair_or_boundary": family.repair,
    }
    if family.status == "EXACT" and any(
        not mapping[field]
        for field in (
            "runtime_entry",
            "code_locations",
            "config_or_schema_locations",
            "regression_test_locations",
            "dynamic_evidence",
        )
    ):
        raise RuntimeError(
            f"EXACT clause lacks specific evidence: {source_name}:{line_number}"
        )
    return mapping


def build_rows(
    source_name: str,
    source: Path,
    project_root: Path,
    include_text: bool,
) -> list[dict[str, str]]:
    spec = SOURCE_SPECS[source_name]
    digest = sha256_file(source)
    if digest != spec["sha256"]:
        raise RuntimeError(
            f"{source_name} SHA-256 changed: expected {spec['sha256']}, observed {digest}"
        )
    rows = []
    section = None
    heading = ""
    source_lines = source.read_text(encoding="utf-8").splitlines()
    for line_index, raw_line in enumerate(source_lines):
        line_number = line_index + 1
        top = TOP_SECTION.match(raw_line)
        if top is not None:
            section = int(top.group(1))
        if section not in TRACKED_SECTIONS:
            continue
        if HEADING.match(raw_line.strip()):
            heading = HEADING.sub("", raw_line.strip())
        table_requirement = bool(
            raw_line.strip().startswith("|")
            and not _is_table_header(source_lines, line_index)
        )
        for clause_index, clause in enumerate(clauses_for_line(raw_line), start=1):
            clause = clause.strip()
            clause_digest = hashlib.sha256(clause.encode("utf-8")).hexdigest()
            signal = normative_signal(
                clause,
                raw_line,
                table_requirement=table_requirement,
            )
            if section == 13 or is_normative_signal(signal):
                mapping = _normative_mapping(
                    source_name,
                    line_number,
                    section,
                    heading,
                    clause,
                    clause_digest,
                    project_root,
                )
            else:
                mapping = _context_mapping(clause_digest)
            rows.append(
                {
                    "requirement_id": (
                        f"{spec['prefix']}-L{line_number:04d}-C{clause_index:02d}"
                    ),
                    "source_document": source_name,
                    "source_document_sha256": digest,
                    "source_line": str(line_number),
                    "clause_index": str(clause_index),
                    "source_clause_sha256": clause_digest,
                    "original_norm": (
                        clause if include_text else "[OMITTED_FROM_PUBLIC_REPOSITORY]"
                    ),
                    "normative_signal": signal,
                    "section": str(section),
                    "section_heading": (
                        heading if include_text else "[OMITTED_FROM_PUBLIC_REPOSITORY]"
                    ),
                    **mapping,
                }
            )
    return rows


def _stage_matrix(path: Path, rows: list[dict[str, str]]) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            descriptor = -1
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def write_matrix_pair(
    public_path: Path,
    public_rows: list[dict[str, str]],
    private_path: Path,
    private_rows: list[dict[str, str]],
) -> None:
    outputs = (public_path, private_path)
    if public_path.resolve(strict=False) == private_path.resolve(strict=False):
        raise ValueError("public and private requirement matrices must use different paths")
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite requirement matrix: {path}")
    for parent in {path.parent for path in outputs}:
        parent.mkdir(parents=True, exist_ok=True)

    staged: list[tuple[Path, Path]] = []
    created: list[Path] = []
    try:
        staged = [
            (_stage_matrix(public_path, public_rows), public_path),
            (_stage_matrix(private_path, private_rows), private_path),
        ]
        for temporary_path, output_path in staged:
            os.link(temporary_path, output_path)
            created.append(output_path)
    except Exception:
        for output_path in reversed(created):
            output_path.unlink(missing_ok=True)
        raise
    finally:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build privacy-safe public and private V6 atomic traceability matrices"
    )
    parser.add_argument("--reader", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--public-output", required=True)
    parser.add_argument("--private-output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    sources = {"reader": Path(args.reader).resolve()}
    public_rows = []
    private_rows = []
    for source_name, source in sources.items():
        public_rows.extend(build_rows(source_name, source, project_root, False))
        private_rows.extend(build_rows(source_name, source, project_root, True))
    if not public_rows or len(public_rows) != len(private_rows):
        raise RuntimeError("V6 requirement extraction produced an invalid row count")
    write_matrix_pair(
        Path(args.public_output),
        public_rows,
        Path(args.private_output),
        private_rows,
    )
    print(f"requirements={len(public_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
