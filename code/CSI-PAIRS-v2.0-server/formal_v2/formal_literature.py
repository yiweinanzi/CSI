from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .formal_evidence import evidence_context
from .formal_io import artifact_manifest, read_strict_json, sha256_file, write_json
from .formal_llm_judge import C13_JUDGE_ATTESTATION, C13_JUDGE_FILENAME, parse_llm_judge


DATABASE_HOSTS = {
    "Crossref": "api.crossref.org",
    "OpenAlex": "api.openalex.org",
    "Semantic Scholar": "api.semanticscholar.org",
}
DATABASE_SEARCH_ENDPOINTS = {
    "Crossref": ("/works", {"query", "query.bibliographic", "query.title"}),
    "OpenAlex": ("/works", {"search"}),
    "Semantic Scholar": ("/graph/v1/paper/search", {"query"}),
}
MANIFEST_SCHEMA = "csi-pairs-v6-literature-resource-manifest-v4"
GATE_SCHEMA = "csi-pairs-v6-literature-resource-gate-v4"


def run_literature_resource_gate(config, dataset, manifest_path, output_root):
    path = Path(manifest_path).resolve()
    manifest = read_strict_json(path)
    review = _validate_manifest(config, manifest, path.parent, dataset)
    output_dir = Path(output_root) / "literature_resources"
    output_dir.mkdir(parents=True, exist_ok=True)
    bound_manifest = output_dir / "literature_manifest.json"
    bound_records = []
    for record in manifest["records"]:
        bound_records.append(
            {
                **record,
                "content_path": str(_resolve_relative(record["content_path"], path.parent)),
            }
        )
    bound_receipts = []
    for receipt in manifest["search_receipts"]:
        bound_receipts.append(
            {
                **receipt,
                "receipt_path": str(_resolve_relative(receipt["receipt_path"], path.parent)),
            }
        )
    write_json(
        bound_manifest,
        {
            **manifest,
            "records": bound_records,
            "search_receipts": bound_receipts,
            "llm_judge_path": str(review["path"]),
        },
    )
    evidence = evidence_context(
        config, dataset, "FORBIDDEN" if dataset.is_fixture else "CANDIDATE_NOT_CLAIM"
    )
    decision = manifest["decision"]
    passed = bool(
        decision["no_direct_overlap"]
        and decision["rt_path_ready"]
        and decision["map_path_ready"]
        and decision["external_validity_path_ready"]
    )
    gate = {
        "schema_version": GATE_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        **evidence,
        "gate": "G0",
        "input_manifest_path": bound_manifest.name,
        "input_manifest_sha256": sha256_file(bound_manifest),
        "search_completed_utc": manifest["search_completed_utc"],
        "databases": manifest["databases"],
        "query_count": len(manifest["queries"]),
        "search_receipt_count": len(manifest["search_receipts"]),
        "search_receipts_verified": True,
        "record_count": len(manifest["records"]),
        "resource_plan": manifest["resource_plan"],
        "decision": decision,
        "c13_requires_llm_judge": True,
        "llm_judge_path": str(review["path"]),
        "llm_judge_sha256": manifest["llm_judge_sha256"],
        "llm_judge": review["judge"],
        "llm_judge_family": review["judge_family"],
        "llm_judge_completed_utc": review["completed_utc"],
        "llm_judge_signature": review["signature"],
        "llm_judge_signed_utc": review["signed_utc"],
    }
    write_json(output_dir / "gate.json", gate)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": "csi-pairs-formal-stage-manifest-v2.1-v6",
            **evidence,
            "files": artifact_manifest(output_dir, evidence=evidence),
        },
    )
    return gate


def _validate_manifest(config, manifest, manifest_root=None, dataset=None):
    required = {
        "schema_version",
        "search_completed_utc",
        "databases",
        "queries",
        "search_receipts",
        "records",
        "resource_plan",
        "licenses_reviewed",
        "decision",
        "llm_judge_path",
        "llm_judge_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("literature/resource manifest fields must be exact")
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise ValueError("literature/resource manifest schema mismatch")
    completed = _parse_utc(manifest["search_completed_utc"])
    age = (datetime.now(timezone.utc) - completed).total_seconds() / 86400.0
    if age < 0 or age > int(config["literature"]["maximum_search_age_days"]):
        raise ValueError("literature search is future-dated or stale")
    databases = manifest["databases"]
    if (
        not isinstance(databases, list)
        or len(databases) != len(set(databases))
        or set(databases) != set(config["literature"]["required_databases"])
        or set(databases).difference(DATABASE_HOSTS)
    ):
        raise ValueError("literature search did not cover the required databases")
    queries = manifest["queries"]
    if (
        not isinstance(queries, list)
        or not queries
        or len(queries) != len(set(queries))
        or any(not isinstance(query, str) or not query.strip() for query in queries)
    ):
        raise ValueError("literature search queries must be unique nonempty strings")
    _validate_receipts(manifest["search_receipts"], databases, queries, completed, manifest_root)
    _validate_records(manifest["records"], manifest_root)

    resource = manifest["resource_plan"]
    resource_fields = {
        "gpu_hours", "storage_gb", "seed_count", "failure_policy", "adapter_owners",
    }
    if not isinstance(resource, dict) or set(resource) != resource_fields:
        raise ValueError("resource plan fields must be exact")
    if float(resource["gpu_hours"]) <= 0 or float(resource["storage_gb"]) <= 0:
        raise ValueError("resource plan compute/storage must be positive")
    if int(resource["seed_count"]) < 3:
        raise ValueError("resource plan must fund at least three seeds")
    if not resource["failure_policy"] or not resource["adapter_owners"]:
        raise ValueError("resource plan ownership/failure policy must be explicit")
    if manifest["licenses_reviewed"] is not True:
        raise ValueError("resource gate requires an explicit license review")
    decision = manifest["decision"]
    decision_fields = {
        "no_direct_overlap", "rt_path_ready", "map_path_ready",
        "external_validity_path_ready", "novelty_scope",
    }
    if not isinstance(decision, dict) or set(decision) != decision_fields:
        raise ValueError("literature/resource decision fields must be exact")
    for key in (
        "no_direct_overlap", "rt_path_ready", "map_path_ready",
        "external_validity_path_ready",
    ):
        if type(decision[key]) is not bool:
            raise ValueError(f"literature/resource decision {key} must be boolean")
    if not isinstance(decision["novelty_scope"], str) or not decision["novelty_scope"].strip():
        raise ValueError("literature novelty scope must be explicit")
    direct_overlap = any(record["relation_to_claim"] == "direct_overlap" for record in manifest["records"])
    if decision["no_direct_overlap"] == direct_overlap:
        raise ValueError("literature direct-overlap decision contradicts its records")
    return _validate_llm_judge_review(
        manifest,
        manifest_root,
        completed,
        dataset,
    )


def _validate_llm_judge_review(manifest, manifest_root, search_completed, dataset):
    digest = manifest["llm_judge_sha256"]
    if not _lower_sha256(digest):
        raise ValueError("C13 llm-judge hash is invalid")
    relative = Path(manifest["llm_judge_path"])
    if relative.name != C13_JUDGE_FILENAME:
        raise ValueError("C13 llm-judge review must be named LLM_JUDGE_REVIEW.md")
    path = _resolve_relative(manifest["llm_judge_path"], manifest_root)
    if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
        raise ValueError("C13 llm-judge review is missing or hash-mismatched")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeError as error:
        raise ValueError("C13 llm-judge review is not valid UTF-8") from error
    forbidden = (
        "TEMPLATE_ONLY_NOT_REVIEWED",
        "<REQUIRED>",
        "<true or false>",
        "<YYYY-MM-DDTHH:MM:SSZ>",
        "<REQUIRED;",
        "I attest that I personally reviewed",
    )
    if len(content.encode("utf-8")) < 512 or any(value in content for value in forbidden):
        raise ValueError("C13 llm-judge review is still a template or is incomplete")

    values = {
        label: _review_field(content, label)
        for label in (
            "Judge family",
            "Judge identity",
            "Authorization basis",
            "Review completed UTC",
            "Project dataset SHA-256",
            "Project source-tree SHA-256",
            "Licenses reviewed for every local PDF/source resource",
            "No direct overlap with the frozen C13 claim",
            "RT path ready",
            "Map path ready",
            "External-validity path ready",
            "Allowed novelty scope",
            "Conflicts or unresolved restrictions",
            "Judge signature or authenticated identity",
            "Signed UTC",
        )
    }
    family, identity = parse_llm_judge(f"{values['Judge family']}:{values['Judge identity']}")
    completed = _parse_utc(values["Review completed UTC"])
    signed = _parse_utc(values["Signed UTC"])
    now = datetime.now(timezone.utc)
    if completed < search_completed or signed < completed or signed > now:
        raise ValueError(
            "C13 llm-judge review must follow the frozen searches and use valid UTC ordering"
        )
    if values["Licenses reviewed for every local PDF/source resource"].lower() != str(
        manifest["licenses_reviewed"]
    ).lower():
        raise ValueError("C13 llm-judge license decision differs from the manifest")
    decision_fields = {
        "No direct overlap with the frozen C13 claim": "no_direct_overlap",
        "RT path ready": "rt_path_ready",
        "Map path ready": "map_path_ready",
        "External-validity path ready": "external_validity_path_ready",
    }
    for label, key in decision_fields.items():
        if values[label].lower() != str(manifest["decision"][key]).lower():
            raise ValueError(f"C13 llm-judge decision differs from the manifest: {key}")
    if values["Allowed novelty scope"] != manifest["decision"]["novelty_scope"]:
        raise ValueError("C13 llm-judge novelty scope differs from the manifest")
    if dataset is not None:
        from .formal_evidence import _source_tree_sha256

        if values["Project dataset SHA-256"] != sha256_file(dataset.source_path):
            raise ValueError("C13 llm-judge review dataset hash mismatch")
        if values["Project source-tree SHA-256"] != _source_tree_sha256():
            raise ValueError("C13 llm-judge review source-tree hash mismatch")
    missing_records = [
        Path(record["content_path"]).name
        for record in manifest["records"]
        if Path(record["content_path"]).name not in content
    ]
    if missing_records:
        raise ValueError(
            f"C13 llm-judge review omits literature records: {missing_records}"
        )
    if C13_JUDGE_ATTESTATION not in content:
        raise ValueError("C13 llm-judge attestation is missing or altered")
    return {
        "path": path,
        "judge": f"{family}:{identity}",
        "judge_family": family,
        "completed_utc": values["Review completed UTC"],
        "signature": values["Judge signature or authenticated identity"],
        "signed_utc": values["Signed UTC"],
    }


def _review_field(content, label):
    prefix = f"- {label}:"
    matches = [line[len(prefix) :].strip() for line in content.splitlines() if line.startswith(prefix)]
    if len(matches) != 1 or not matches[0]:
        raise ValueError(f"C13 llm-judge field is missing or duplicated: {label}")
    return matches[0].strip("`")


def _validate_receipts(receipts, databases, queries, completed, manifest_root):
    fields = {
        "database", "query", "searched_utc", "retrieval_url", "result_count",
        "receipt_path", "receipt_sha256",
    }
    expected_pairs = {(database, query) for database in databases for query in queries}
    if not isinstance(receipts, list) or len(receipts) != len(expected_pairs):
        raise ValueError("literature search requires one receipt for every database/query pair")
    observed = set()
    for receipt in receipts:
        if not isinstance(receipt, dict) or set(receipt) != fields:
            raise ValueError("literature search receipt fields must be exact")
        pair = (receipt["database"], receipt["query"])
        if pair not in expected_pairs or pair in observed:
            raise ValueError("literature search receipts have missing, duplicate, or unexpected coverage")
        observed.add(pair)
        searched = _parse_utc(receipt["searched_utc"])
        if searched > completed or (completed - searched).total_seconds() > 86400:
            raise ValueError("literature receipt time is after or too far before search completion")
        parsed = _valid_https_url(receipt["retrieval_url"])
        if parsed.hostname != DATABASE_HOSTS[receipt["database"]]:
            raise ValueError("literature receipt retrieval URL is not an approved database API")
        endpoint, query_keys = DATABASE_SEARCH_ENDPOINTS[receipt["database"]]
        parameters = parse_qs(parsed.query, keep_blank_values=True)
        if parsed.path.rstrip("/") != endpoint or not any(
            receipt["query"] in parameters.get(key, []) for key in query_keys
        ):
            raise ValueError(
                "literature receipt retrieval URL does not bind the frozen query"
            )
        count = receipt["result_count"]
        if type(count) is not int or count < 1:
            raise ValueError("literature receipt result_count must be positive")
        digest = receipt["receipt_sha256"]
        if not _lower_sha256(digest):
            raise ValueError("literature receipt hash is invalid")
        path = _resolve_relative(receipt["receipt_path"], manifest_root)
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise ValueError("literature search receipt is missing or hash-mismatched")
        payload = read_strict_json(path)
        results = _receipt_results(receipt["database"], payload)
        if len(results) != count or any(not _result_has_identity(receipt["database"], row) for row in results):
            raise ValueError("literature search receipt result count or identities are invalid")
    if observed != expected_pairs:
        raise ValueError("literature search receipt coverage is incomplete")


def _validate_records(records, manifest_root):
    record_fields = {
        "citation_key", "title", "doi_or_url", "verified_utc", "content_path", "content_sha256",
        "relation_to_claim", "implementation_status",
    }
    if not isinstance(records, list) or not records:
        raise ValueError("literature manifest requires verified records")
    if len({record.get("citation_key") for record in records if isinstance(record, dict)}) != len(records):
        raise ValueError("literature citation keys must be unique")
    for record in records:
        if not isinstance(record, dict) or set(record) != record_fields:
            raise ValueError("literature record fields must be exact")
        if any(not isinstance(record[key], str) or not record[key].strip() for key in record_fields):
            raise ValueError("literature record values must be nonempty strings")
        _valid_https_url(record["doi_or_url"])
        digest = record["content_sha256"]
        if not _lower_sha256(digest):
            raise ValueError("literature record content hash is invalid")
        _parse_utc(record["verified_utc"])
        if record["relation_to_claim"] not in {
            "direct_overlap", "adjacent_nonoverlap", "baseline", "facility",
        }:
            raise ValueError("literature relation_to_claim is not a frozen category")
        if record["implementation_status"] not in {
            "integrated", "adapter_ready", "paper_only", "unavailable",
        }:
            raise ValueError("literature implementation_status is not a frozen category")
        content = _resolve_relative(record["content_path"], manifest_root)
        if (
            not content.is_file()
            or content.is_symlink()
            or sha256_file(content) != digest
            or not content.read_bytes().startswith(b"%PDF-")
        ):
            raise ValueError("literature content is missing, hash-mismatched, or not a PDF")


def _receipt_results(database, payload):
    if database == "Crossref":
        return payload.get("message", {}).get("items", []) if isinstance(payload, dict) else []
    if database == "OpenAlex":
        return payload.get("results", []) if isinstance(payload, dict) else []
    if database == "Semantic Scholar":
        return payload.get("data", []) if isinstance(payload, dict) else []
    return []


def _result_has_identity(database, row):
    if not isinstance(row, dict):
        return False
    if database == "Crossref":
        return isinstance(row.get("DOI"), str) and bool(row["DOI"].strip())
    if database == "OpenAlex":
        return isinstance(row.get("id"), str) and bool(row["id"].strip())
    if database == "Semantic Scholar":
        return isinstance(row.get("paperId"), str) and bool(row["paperId"].strip())
    return False


def _valid_https_url(value):
    if not isinstance(value, str):
        raise ValueError("literature URL must be a string")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("literature URL must be a credential-free HTTPS URL")
    return parsed


def _resolve_relative(path_value, manifest_root):
    path = Path(path_value)
    if not path.is_absolute():
        if manifest_root is None:
            raise ValueError("relative literature path requires a manifest root")
        path = Path(manifest_root) / path
    if path.is_symlink():
        raise ValueError("literature inputs must be regular files, not symlinks")
    return path.resolve()


def _lower_sha256(value):
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _parse_utc(value):
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("timestamps must use UTC Z format")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("invalid UTC timestamp") from error
