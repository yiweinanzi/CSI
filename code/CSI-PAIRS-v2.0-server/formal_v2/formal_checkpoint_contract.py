from __future__ import annotations

from .formal_evidence import EVIDENCE_AUTH_KEYS


FORMAL_CHECKPOINT_SCHEMA = "csi-pairs-formal-checkpoint-v2.1-v6"
CHECKPOINT_EVIDENCE_FIELDS = frozenset((*EVIDENCE_AUTH_KEYS, "scientific_use"))
FORMAL_CHECKPOINT_FIELDS = frozenset(
    {
        "schema_version",
        "arm",
        "seed",
        "model_spec",
        "normalization",
        "teacher_checkpoint_sha256",
        "checkpoint_rule",
        "state_dict",
    }
) | CHECKPOINT_EVIDENCE_FIELDS

SHUFFLED_CHECKPOINT_SCHEMA = "csi-pairs-v6-shuffled-formal-checkpoint-v1"
SHUFFLED_CHECKPOINT_FIELDS = FORMAL_CHECKPOINT_FIELDS | frozenset(
    {"pairing_breaks", "training_provenance_sha256"}
)
