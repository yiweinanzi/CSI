from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from numbers import Integral
from typing import Iterable, Iterator, Mapping

import numpy as np

from .formal_protocol import PatchSpec


COMPATIBILITY_PAIR_TABLE = "compatibility_pair_effects.csv"
RESPONSE_PAIR_TABLE = "response_pair_effects.csv"
PAIR_TABLES = (COMPATIBILITY_PAIR_TABLE, RESPONSE_PAIR_TABLE)

COMPATIBILITY_IDENTITY_SCHEMA = (
    "csi-pairs-v6-evaluation-compatibility-pair-identity-v1"
)
RESPONSE_IDENTITY_SCHEMA = "csi-pairs-v6-evaluation-response-pair-identity-v1"


class PairInventoryError(RuntimeError):
    """Raised when a dataset or pair row violates the frozen evaluation domain."""


@dataclass(frozen=True)
class PairInventorySummary:
    schema: str
    row_count: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "row_count": self.row_count,
            "sha256": self.sha256,
        }


def _canonical_table(table: object) -> str:
    if table == COMPATIBILITY_PAIR_TABLE or table == "compatibility_pair_effects":
        return COMPATIBILITY_PAIR_TABLE
    if table == RESPONSE_PAIR_TABLE or table == "response_pair_effects":
        return RESPONSE_PAIR_TABLE
    raise PairInventoryError(f"unsupported evaluation pair table: {table!r}")


def _table_schema(table: str) -> str:
    if table == COMPATIBILITY_PAIR_TABLE:
        return COMPATIBILITY_IDENTITY_SCHEMA
    if table == RESPONSE_PAIR_TABLE:
        return RESPONSE_IDENTITY_SCHEMA
    raise AssertionError(f"uncanonicalized pair table: {table!r}")


def _canonical_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise PairInventoryError(f"pair identity integer is invalid: {field}")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value, 10)
        except ValueError as error:
            raise PairInventoryError(
                f"pair identity integer is invalid: {field}"
            ) from error
        if str(parsed) != value:
            raise PairInventoryError(
                f"pair identity integer is non-canonical: {field}"
            )
        return parsed
    raise PairInventoryError(f"pair identity integer is invalid: {field}")


def _canonical_scene(scene: object) -> int:
    value = _canonical_integer(scene, field="scene_index")
    if value < 0:
        raise PairInventoryError("evaluation scene index must be nonnegative")
    return value


def _canonical_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise PairInventoryError(f"pair identity text is invalid: {field}")
    return str(value)


_MISSING = object()


def _edge_value(edge: object, field: str, default: object = _MISSING) -> object:
    if isinstance(edge, Mapping):
        value = edge.get(field, default)
    else:
        value = getattr(edge, field, default)
    if value is _MISSING:
        raise PairInventoryError(f"dataset directed edge field is missing: {field}")
    return value


def _encode_identity(record: Mapping[str, object]) -> bytes:
    try:
        encoded = json.dumps(
            dict(record),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, UnicodeError, ValueError) as error:
        raise PairInventoryError("pair identity record is not canonical JSON") from error
    return encoded + b"\n"


def _lexical_component(value: int, *, followed_by_component: bool) -> str:
    suffix = ":" if followed_by_component else ""
    return f"{value}{suffix}"


@dataclass(frozen=True)
class ScenePairUniverse:
    """Finite legacy pair domain for one evaluation scene, without CSI payloads."""

    scene_index: int
    bank_id: str
    scene_role: str
    world_count: int
    bit_count: int
    position_count: int
    query_count: int
    natural_world_index: int
    directed_edges: frozenset[tuple[int, int]]
    headline_directed_edges: frozenset[tuple[int, int]]
    eligible_positions: frozenset[int]
    compatibility_inventory: PairInventorySummary = field(init=False)
    response_inventory: PairInventorySummary = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "compatibility_inventory",
            _summarize_records(
                COMPATIBILITY_PAIR_TABLE,
                self.iter_expected_records(COMPATIBILITY_PAIR_TABLE),
            ),
        )
        object.__setattr__(
            self,
            "response_inventory",
            _summarize_records(
                RESPONSE_PAIR_TABLE,
                self.iter_expected_records(RESPONSE_PAIR_TABLE),
            ),
        )

    def summary(self, table: object) -> dict[str, object]:
        canonical = _canonical_table(table)
        value = (
            self.compatibility_inventory
            if canonical == COMPATIBILITY_PAIR_TABLE
            else self.response_inventory
        )
        return value.as_dict()

    @property
    def compatibility_pair_effects(self) -> dict[str, object]:
        return self.compatibility_inventory.as_dict()

    @property
    def response_pair_effects(self) -> dict[str, object]:
        return self.response_inventory.as_dict()

    def expected_row_count(self, table: object) -> int:
        return int(self.summary(table)["row_count"])

    def iter_expected_records(self, table: object) -> Iterator[dict[str, object]]:
        canonical = _canonical_table(table)
        if canonical == COMPATIBILITY_PAIR_TABLE:
            yield from self._iter_compatibility_records()
        else:
            yield from self._iter_response_records()

    def validator(self, table: object) -> "PairRowValidator":
        return PairRowValidator(self, table)

    def _iter_compatibility_records(self) -> Iterator[dict[str, object]]:
        undirected_edges = {
            (min(source, target), max(source, target))
            for source, target in self.headline_directed_edges
        }
        ordered_edges = sorted(
            undirected_edges,
            key=lambda edge: f"{edge[0]}:{edge[1]}:",
        )
        ordered_positions = sorted(
            self.eligible_positions,
            key=lambda position: _lexical_component(
                position, followed_by_component=True
            ),
        )
        for first, second in ordered_edges:
            for position in ordered_positions:
                for source in sorted((first, second), key=lambda value: str(value)):
                    target = second if source == first else first
                    pair_id = (
                        f"{self.bank_id}:{first}:{second}:{position}:{source}"
                    )
                    yield {
                        "pair_id": pair_id,
                        "scene_index": self.scene_index,
                        "bank_id": self.bank_id,
                        "source_world": source,
                        "target_world": target,
                        "position_index": position,
                    }

    def _iter_response_records(self) -> Iterator[dict[str, object]]:
        ordered_edges = sorted(
            self.directed_edges,
            key=lambda edge: f"{edge[0]}:{edge[1]}:",
        )
        ordered_positions = sorted(
            self.eligible_positions,
            key=lambda position: _lexical_component(
                position, followed_by_component=True
            ),
        )
        ordered_queries = sorted(range(self.query_count), key=lambda query: str(query))
        for source, target in ordered_edges:
            for position in ordered_positions:
                for query in ordered_queries:
                    pair_id = (
                        f"{self.bank_id}:{source}:{target}:{position}:{query}"
                    )
                    yield {
                        "pair_id": pair_id,
                        "scene_index": self.scene_index,
                        "bank_id": self.bank_id,
                        "source_world": source,
                        "target_world": target,
                        "position_index": position,
                        "query_index": query,
                    }


def _summarize_records(
    table: str, records: Iterable[Mapping[str, object]]
) -> PairInventorySummary:
    digest = hashlib.sha256()
    count = 0
    previous_pair_id: str | None = None
    for record in records:
        pair_id = _canonical_text(record.get("pair_id"), field="pair_id")
        if previous_pair_id is not None and pair_id <= previous_pair_id:
            raise PairInventoryError(
                f"expected pair universe is not strictly increasing: {table}"
            )
        previous_pair_id = pair_id
        digest.update(_encode_identity(record))
        count += 1
    return PairInventorySummary(
        schema=_table_schema(table),
        row_count=count,
        sha256=digest.hexdigest(),
    )


def _metadata_arrays(dataset: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        bank_ids = np.asarray(dataset.bank_ids)
        scene_roles = np.asarray(dataset.scene_roles)
        position_roles = np.asarray(dataset.position_roles)
        natural_world_indices = np.asarray(dataset.natural_world_index)
    except (AttributeError, TypeError, ValueError) as error:
        raise PairInventoryError("dataset evaluation identity metadata is missing") from error
    if bank_ids.ndim != 1 or scene_roles.shape != bank_ids.shape:
        raise PairInventoryError("dataset scene/bank metadata shape is invalid")
    if (
        position_roles.ndim != 2
        or position_roles.shape[0] != bank_ids.size
        or natural_world_indices.shape != bank_ids.shape
    ):
        raise PairInventoryError("dataset scene/position metadata shape is invalid")
    return bank_ids, scene_roles, position_roles, natural_world_indices


def _world_edge_universe(
    dataset: object, scene: int
) -> tuple[int, int, frozenset[tuple[int, int]], tuple[object, ...]]:
    try:
        bits_array = np.asarray(dataset.world_bits)
    except (AttributeError, TypeError, ValueError) as error:
        raise PairInventoryError("dataset world-bit metadata is missing") from error
    if bits_array.ndim != 2 or bits_array.shape[0] < 2 or bits_array.shape[1] < 1:
        raise PairInventoryError("dataset world_bits must be a nontrivial matrix")
    world_count, bit_count = (int(value) for value in bits_array.shape)
    if world_count != 2**bit_count:
        raise PairInventoryError("dataset world_bits is not a complete binary hypercube")
    try:
        rows = tuple(
            tuple(_canonical_integer(value, field="world_bits") for value in row)
            for row in bits_array
        )
    except TypeError as error:
        raise PairInventoryError("dataset world_bits rows are invalid") from error
    if any(value not in {0, 1} for row in rows for value in row):
        raise PairInventoryError("dataset world_bits must be binary")
    lookup = {row: index for index, row in enumerate(rows)}
    if len(lookup) != world_count:
        raise PairInventoryError("dataset world_bits rows must be unique")

    expected: set[tuple[int, int, int]] = set()
    for source, bits in enumerate(rows):
        for bit_index in range(bit_count):
            target_bits = list(bits)
            target_bits[bit_index] = 1 - target_bits[bit_index]
            target = lookup.get(tuple(target_bits))
            if target is None:
                raise PairInventoryError(
                    "dataset world_bits omits a one-bit hypercube neighbor"
                )
            expected.add((source, target, bit_index))

    try:
        provided_edges = tuple(dataset.directed_edges(scene))
    except (AttributeError, TypeError, ValueError, KeyError) as error:
        raise PairInventoryError("dataset directed edge enumeration failed") from error
    observed: set[tuple[int, int, int]] = set()
    for edge in provided_edges:
        try:
            edge_scene_value = _edge_value(edge, "scene", scene)
            edge_scene = _canonical_integer(edge_scene_value, field="edge.scene")
            source = _canonical_integer(
                _edge_value(edge, "source_world"), field="edge.source_world"
            )
            target = _canonical_integer(
                _edge_value(edge, "target_world"), field="edge.target_world"
            )
            bit_index = _canonical_integer(
                _edge_value(edge, "bit_index"), field="edge.bit_index"
            )
        except PairInventoryError:
            raise
        except (AttributeError, TypeError) as error:
            raise PairInventoryError("dataset directed edge fields are incomplete") from error
        identity = (source, target, bit_index)
        if edge_scene != scene or identity not in expected:
            raise PairInventoryError("dataset contains an illegal directed one-bit edge")
        if identity in observed:
            raise PairInventoryError("dataset contains a duplicate directed one-bit edge")
        observed.add(identity)
    if observed != expected:
        raise PairInventoryError("dataset directed one-bit edge universe is incomplete")
    return (
        world_count,
        bit_count,
        frozenset((source, target) for source, target, _bit in observed),
        provided_edges,
    )


def _dataset_query_count(dataset: object) -> int:
    """Read the frozen patch cardinality without touching any CSI array."""
    metadata = getattr(dataset, "metadata", None)
    representation = metadata.get("representation") if isinstance(metadata, Mapping) else None
    if isinstance(representation, Mapping) and "patch_count" in representation:
        try:
            value = _canonical_integer(representation["patch_count"], field="patch_count")
        except PairInventoryError:
            raise
        except (TypeError, ValueError) as error:
            raise PairInventoryError("dataset patch/query metadata is invalid") from error
        if value < 1:
            raise PairInventoryError("evaluation query universe is empty")
        return value
    if isinstance(dataset, Mapping) and "query_count" in dataset:
        value = _canonical_integer(dataset["query_count"], field="query_count")
        if value < 1:
            raise PairInventoryError("evaluation query universe is empty")
        return value
    direct = getattr(dataset, "query_count", None)
    if direct is not None:
        value = _canonical_integer(direct, field="query_count")
        if value < 1:
            raise PairInventoryError("evaluation query universe is empty")
        return value
    try:
        value = int(PatchSpec.from_metadata(metadata).patch_count)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise PairInventoryError("dataset patch/query metadata is invalid") from error
    if value < 1:
        raise PairInventoryError("evaluation query universe is empty")
    return value


def build_scene_pair_universe(dataset: object, scene: object) -> ScenePairUniverse:
    """Rebuild both legacy pair tables from dataset metadata only."""

    scene_index = _canonical_scene(scene)
    bank_ids, scene_roles, position_roles, natural_indices = _metadata_arrays(dataset)
    if scene_index >= bank_ids.size:
        raise PairInventoryError("evaluation scene index is outside the dataset")
    bank_id = _canonical_text(bank_ids[scene_index], field="bank_id")
    scene_role = _canonical_text(scene_roles[scene_index], field="scene_role")
    if scene_role not in {"source_final_unseen_bank", "target"}:
        raise PairInventoryError(
            f"unsupported evaluation scene role: {scene_role!r}"
        )

    position_count = int(position_roles.shape[1])
    if position_count < 1:
        raise PairInventoryError("evaluation scene has no positions")
    if scene_role == "target":
        eligible_positions = frozenset(
            int(value)
            for value in np.flatnonzero(position_roles[scene_index] == "query")
        )
        if not eligible_positions:
            raise PairInventoryError(
                "target evaluation has no query positions after support exclusion"
            )
    else:
        eligible_positions = frozenset(range(position_count))

    world_count, bit_count, directed_edges, provided_edges = _world_edge_universe(
        dataset, scene_index
    )
    natural = _canonical_integer(
        natural_indices[scene_index], field="natural_world_index"
    )
    if natural < 0 or natural >= world_count:
        raise PairInventoryError("dataset natural world is outside the world universe")

    headline_directed_edges: set[tuple[int, int]] = set()
    for edge in provided_edges:
        source = _canonical_integer(
            _edge_value(edge, "source_world"), field="edge.source_world"
        )
        target = _canonical_integer(
            _edge_value(edge, "target_world"), field="edge.target_world"
        )
        expected_eligible = source != natural and target != natural
        if expected_eligible:
            headline_directed_edges.add((source, target))

    query_count = _dataset_query_count(dataset)

    return ScenePairUniverse(
        scene_index=scene_index,
        bank_id=bank_id,
        scene_role=scene_role,
        world_count=world_count,
        bit_count=bit_count,
        position_count=position_count,
        query_count=query_count,
        natural_world_index=natural,
        directed_edges=directed_edges,
        headline_directed_edges=frozenset(headline_directed_edges),
        eligible_positions=eligible_positions,
    )


class PairRowValidator:
    """Streaming accumulator that proves one complete canonical scene pair table."""

    def __init__(self, universe: ScenePairUniverse, table: object):
        if not isinstance(universe, ScenePairUniverse):
            raise TypeError("pair row validator requires a ScenePairUniverse")
        self.universe = universe
        self.table = _canonical_table(table)
        self._expected = universe.summary(self.table)
        self._digest = hashlib.sha256()
        self._row_count = 0
        self._previous_pair_id: str | None = None
        self._finished: dict[str, object] | None = None
        self._failed = False

    @property
    def row_count(self) -> int:
        return self._row_count

    def feed(self, row: Mapping[str, object]) -> dict[str, object]:
        if self._failed:
            raise PairInventoryError("pair row validator is in a failed state")
        if self._finished is not None:
            raise PairInventoryError("cannot feed a finished pair row validator")
        try:
            record = self._canonical_record(row)
            pair_id = str(record["pair_id"])
            if self._previous_pair_id is not None and pair_id <= self._previous_pair_id:
                raise PairInventoryError(
                    f"pair IDs are not strictly increasing and unique: {self.table}"
                )
            if self._row_count >= int(self._expected["row_count"]):
                raise PairInventoryError(
                    f"pair row count exceeds the dataset universe: {self.table}"
                )
            encoded = _encode_identity(record)
        except Exception:
            self._failed = True
            raise
        self._digest.update(encoded)
        self._row_count += 1
        self._previous_pair_id = pair_id
        return record

    def finish(self) -> dict[str, object]:
        if self._failed:
            raise PairInventoryError("pair row validator is in a failed state")
        if self._finished is not None:
            return dict(self._finished)
        expected_count = int(self._expected["row_count"])
        if self._row_count != expected_count:
            self._failed = True
            raise PairInventoryError(
                f"pair row universe is incomplete: {self.table}; "
                f"expected {expected_count}, observed {self._row_count}"
            )
        summary = {
            "schema": _table_schema(self.table),
            "row_count": self._row_count,
            "sha256": self._digest.hexdigest(),
        }
        if summary != self._expected:
            self._failed = True
            raise PairInventoryError(
                f"pair row universe digest differs from the dataset: {self.table}"
            )
        self._finished = dict(summary)
        return summary

    def _canonical_record(
        self, row: Mapping[str, object]
    ) -> dict[str, object]:
        if not isinstance(row, Mapping):
            raise PairInventoryError("evaluation pair row must be a mapping")
        required = {
            "pair_id",
            "scene_index",
            "bank_id",
            "source_world",
            "target_world",
            "position_index",
        }
        if self.table == RESPONSE_PAIR_TABLE:
            required.add("query_index")
        missing = required.difference(row)
        if missing:
            raise PairInventoryError(
                f"evaluation pair row identity fields are missing: {sorted(missing)}"
            )

        scene = _canonical_integer(row["scene_index"], field="scene_index")
        bank = _canonical_text(row["bank_id"], field="bank_id")
        source = _canonical_integer(row["source_world"], field="source_world")
        target = _canonical_integer(row["target_world"], field="target_world")
        position = _canonical_integer(row["position_index"], field="position_index")
        pair_id = _canonical_text(row["pair_id"], field="pair_id")
        if scene != self.universe.scene_index:
            raise PairInventoryError("pair row scene does not match the dataset scene")
        if bank != self.universe.bank_id:
            raise PairInventoryError("pair row bank does not match the dataset bank")
        if position not in self.universe.eligible_positions:
            raise PairInventoryError("pair row position is outside the evaluation positions")

        record: dict[str, object] = {
            "pair_id": pair_id,
            "scene_index": scene,
            "bank_id": bank,
            "source_world": source,
            "target_world": target,
            "position_index": position,
        }
        if self.table == COMPATIBILITY_PAIR_TABLE:
            if (source, target) not in self.universe.headline_directed_edges:
                if (
                    (source, target) in self.universe.directed_edges
                    and (
                        source
                        == target
                        or source == self.universe.natural_world_index
                        or target == self.universe.natural_world_index
                    )
                ):
                    raise PairInventoryError(
                        "compatibility pair uses a natural-incident headline edge"
                    )
                raise PairInventoryError(
                    "compatibility pair is not a headline directed one-bit edge"
                )
            first, second = sorted((source, target))
            expected_pair_id = (
                f"{bank}:{first}:{second}:{position}:{source}"
            )
        else:
            if (source, target) not in self.universe.directed_edges:
                raise PairInventoryError("response pair is not a directed one-bit edge")
            query = _canonical_integer(row["query_index"], field="query_index")
            if query < 0 or query >= self.universe.query_count:
                raise PairInventoryError("response pair query is outside the patch bounds")
            record["query_index"] = query
            expected_pair_id = f"{bank}:{source}:{target}:{position}:{query}"
        if pair_id != expected_pair_id:
            raise PairInventoryError(
                f"pair_id does not encode the complete dataset identity: {self.table}"
            )
        return record

def create_pair_row_validator(
    universe: ScenePairUniverse, table: object
) -> PairRowValidator:
    return PairRowValidator(universe, table)


# Descriptive alias for callers that prefer the accumulator terminology.
PairRowValidatorAccumulator = PairRowValidator
StreamingPairRowValidatorAccumulator = PairRowValidator
PairInventoryAccumulator = PairRowValidator


__all__ = [
    "COMPATIBILITY_PAIR_TABLE",
    "RESPONSE_PAIR_TABLE",
    "PAIR_TABLES",
    "PairInventoryError",
    "PairInventorySummary",
    "ScenePairUniverse",
    "PairRowValidator",
    "PairRowValidatorAccumulator",
    "StreamingPairRowValidatorAccumulator",
    "PairInventoryAccumulator",
    "build_scene_pair_universe",
    "create_pair_row_validator",
]
