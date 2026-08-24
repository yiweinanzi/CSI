from __future__ import annotations


LLM_JUDGE_FAMILIES = ("codex", "claude-code", "cursor")
C13_JUDGE_FILENAME = "LLM_JUDGE_REVIEW.md"
LLM_JUDGE_APPROVAL_SCHEMA = "csi-pairs-full-run-llm-judge-approval-v1"
AWAITING_LLM_JUDGE = "AWAITING_LLM_JUDGE"
LLM_JUDGE_REQUIRED = "LLM_JUDGE_REQUIRED"
C13_JUDGE_ATTESTATION = (
    "This LLM-as-judge review bound the listed resources and the frozen C13\n"
    "claim, verified the recorded license/redistribution decisions from the cited\n"
    "sources, and made the novelty and readiness decisions above.\n"
    "The judge family is one of: codex, claude-code, cursor."
)
APPROVAL_ATTESTATION = (
    "This LLM-as-judge (codex, claude-code, or cursor) reviewed the bound G0, "
    "independent RT, G1/G2, Response-control, G8 independent external-validity, "
    "input, runtime, and compute-plan evidence and authorizes only this run nonce."
)


def parse_llm_judge(value: str) -> tuple[str, str]:
    if not isinstance(value, str) or value.strip() != value or ":" not in value:
        raise ValueError(
            "llm-judge identity must be nonempty family:identity "
            f"with family in {LLM_JUDGE_FAMILIES}"
        )
    family, identity = value.split(":", 1)
    if family not in LLM_JUDGE_FAMILIES or not identity.strip() or identity.strip() != identity:
        raise ValueError(
            "llm-judge family must be one of "
            f"{LLM_JUDGE_FAMILIES} and identity must be nonempty"
        )
    if identity.lower() in {"human", "authorized-human", "reviewer"}:
        raise ValueError("llm-judge identity cannot be a human-review placeholder")
    return family, identity
