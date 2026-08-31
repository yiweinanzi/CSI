from __future__ import annotations

import unittest
from dataclasses import dataclass

import numpy as np

from formal_v2.formal_evaluation_inventory import (
    COMPATIBILITY_PAIR_TABLE,
    RESPONSE_PAIR_TABLE,
    PairInventoryError,
    ScenePairUniverse,
    build_scene_pair_universe,
)


@dataclass(frozen=True)
class _Edge:
    scene: int
    source_world: int
    target_world: int
    bit_index: int


class _FakeDataset:
    world_bits = np.asarray(
        (
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ),
        dtype=np.int64,
    )
    bank_ids = np.asarray(("source-bank", "city:bank:alpha"))
    scene_roles = np.asarray(("source_final_unseen_bank", "target"))
    position_roles = np.asarray(
        (
            ("standard",) * 12,
            (
                "support_pool",
                "query",
                "query",
                "support_pool",
                "support_pool",
                "support_pool",
                "support_pool",
                "support_pool",
                "support_pool",
                "support_pool",
                "query",
                "query",
            ),
        )
    )
    natural_world_index = np.asarray((0, 3), dtype=np.int64)
    metadata = {
        "representation": {
            "antenna_count": 3,
            "subcarrier_count": 4,
            "patch_complex_size": 1,
            "patch_antenna_size": 1,
            "patch_subcarrier_size": 1,
        }
    }

    @property
    def csi_repeat(self):
        raise AssertionError("pair inventory must not inspect CSI")

    @property
    def csi(self):
        raise AssertionError("pair inventory must not inspect CSI")

    @property
    def world_count(self):
        raise AssertionError("pair inventory must derive worlds from world_bits")

    @property
    def position_count(self):
        raise AssertionError("pair inventory must derive positions from role metadata")

    def directed_edges(self, scene):
        lookup = {
            tuple(int(value) for value in row): index
            for index, row in enumerate(self.world_bits)
        }
        for source, bits in enumerate(self.world_bits):
            for bit_index in range(self.world_bits.shape[1]):
                target_bits = bits.copy()
                target_bits[bit_index] = 1 - target_bits[bit_index]
                yield _Edge(
                    scene=int(scene),
                    source_world=source,
                    target_world=lookup[tuple(int(value) for value in target_bits)],
                    bit_index=bit_index,
                )


def _persisted(record):
    return {
        key: str(value) if isinstance(value, int) else value
        for key, value in record.items()
    }


def _legacy_records(dataset, scene, table):
    scene = int(scene)
    bank = str(dataset.bank_ids[scene])
    positions = (
        np.flatnonzero(dataset.position_roles[scene] == "query").tolist()
        if str(dataset.scene_roles[scene]) == "target"
        else list(range(dataset.position_roles.shape[1]))
    )
    records = []
    if table == COMPATIBILITY_PAIR_TABLE:
        natural = int(dataset.natural_world_index[scene])
        for edge in dataset.directed_edges(scene):
            if edge.source_world >= edge.target_world:
                continue
            if edge.source_world == natural or edge.target_world == natural:
                continue
            for position in positions:
                for source in (edge.source_world, edge.target_world):
                    target = (
                        edge.target_world
                        if source == edge.source_world
                        else edge.source_world
                    )
                    records.append(
                        {
                            "pair_id": (
                                f"{bank}:{edge.source_world}:{edge.target_world}:"
                                f"{position}:{source}"
                            ),
                            "scene_index": scene,
                            "bank_id": bank,
                            "source_world": source,
                            "target_world": target,
                            "position_index": position,
                        }
                    )
    else:
        query_count = 12
        for edge in dataset.directed_edges(scene):
            for position in positions:
                for query in range(query_count):
                    records.append(
                        {
                            "pair_id": (
                                f"{bank}:{edge.source_world}:{edge.target_world}:"
                                f"{position}:{query}"
                            ),
                            "scene_index": scene,
                            "bank_id": bank,
                            "source_world": edge.source_world,
                            "target_world": edge.target_world,
                            "position_index": position,
                            "query_index": query,
                        }
                    )
    return sorted(records, key=lambda row: row["pair_id"])


class ScenePairUniverseTests(unittest.TestCase):
    def setUp(self):
        self.dataset = _FakeDataset()
        self.universe = build_scene_pair_universe(self.dataset, 1)

    def test_build_matches_legacy_domains_without_reading_csi(self):
        self.assertIsInstance(self.universe, ScenePairUniverse)
        self.assertEqual(self.universe.bank_id, "city:bank:alpha")
        self.assertEqual(self.universe.eligible_positions, frozenset((1, 2, 10, 11)))
        self.assertEqual(self.universe.query_count, 12)
        self.assertEqual(
            list(self.universe.iter_expected_records(COMPATIBILITY_PAIR_TABLE)),
            _legacy_records(self.dataset, 1, COMPATIBILITY_PAIR_TABLE),
        )
        self.assertEqual(
            list(self.universe.iter_expected_records(RESPONSE_PAIR_TABLE)),
            _legacy_records(self.dataset, 1, RESPONSE_PAIR_TABLE),
        )
        self.assertEqual(self.universe.expected_row_count(COMPATIBILITY_PAIR_TABLE), 16)
        self.assertEqual(self.universe.expected_row_count(RESPONSE_PAIR_TABLE), 384)
        self.assertEqual(
            self.universe.summary(COMPATIBILITY_PAIR_TABLE)["sha256"],
            "31062d46ed61e4c94ba52b9254c6c2c0dd0638bbf6336ae87cebbe17e3356a04",
        )
        self.assertEqual(
            self.universe.summary(RESPONSE_PAIR_TABLE)["sha256"],
            "da0efb0f4f6fd29c3f2bdb058246a158218d41f64352ab5a1c2509af779f30fd",
        )
        compatibility = list(
            self.universe.iter_expected_records(COMPATIBILITY_PAIR_TABLE)
        )
        self.assertTrue(
            all(
                3 not in (row["source_world"], row["target_world"])
                for row in compatibility
            )
        )
        response = list(self.universe.iter_expected_records(RESPONSE_PAIR_TABLE))
        self.assertEqual({row["query_index"] for row in response}, set(range(12)))

    def test_source_evaluation_uses_every_position(self):
        source = build_scene_pair_universe(self.dataset, 0)
        self.assertEqual(source.eligible_positions, frozenset(range(12)))
        self.assertEqual(source.expected_row_count(COMPATIBILITY_PAIR_TABLE), 48)
        self.assertEqual(source.expected_row_count(RESPONSE_PAIR_TABLE), 1152)

    def test_typed_and_persisted_rows_have_the_same_canonical_summary(self):
        for table in (COMPATIBILITY_PAIR_TABLE, RESPONSE_PAIR_TABLE):
            expected = self.universe.summary(table)
            self.assertEqual(set(expected), {"schema", "row_count", "sha256"})
            self.assertEqual(len(expected["sha256"]), 64)

            typed = self.universe.validator(table)
            persisted = self.universe.validator(table.removesuffix(".csv"))
            records = list(self.universe.iter_expected_records(table))
            for record in records:
                typed.feed(record)
                persisted.feed(_persisted(record))
            self.assertEqual(typed.finish(), expected)
            self.assertEqual(persisted.finish(), expected)
            self.assertEqual(typed.finish(), expected)

    def test_pair_ids_are_unique_strict_legacy_text_order(self):
        for table in (COMPATIBILITY_PAIR_TABLE, RESPONSE_PAIR_TABLE):
            pair_ids = [
                row["pair_id"]
                for row in self.universe.iter_expected_records(table)
            ]
            self.assertEqual(pair_ids, sorted(pair_ids))
            self.assertEqual(len(pair_ids), len(set(pair_ids)))
        response_ids = [
            row["pair_id"]
            for row in self.universe.iter_expected_records(RESPONSE_PAIR_TABLE)
        ]
        self.assertLess(
            response_ids.index("city:bank:alpha:0:1:10:0"),
            response_ids.index("city:bank:alpha:0:1:1:0"),
        )
        self.assertLess(
            response_ids.index("city:bank:alpha:0:1:1:10"),
            response_ids.index("city:bank:alpha:0:1:1:2"),
        )

    def test_missing_duplicate_and_out_of_order_rows_fail_closed(self):
        records = list(
            self.universe.iter_expected_records(COMPATIBILITY_PAIR_TABLE)
        )
        missing = self.universe.validator(COMPATIBILITY_PAIR_TABLE)
        for row in records[:-1]:
            missing.feed(row)
        with self.assertRaisesRegex(PairInventoryError, "incomplete"):
            missing.finish()

        duplicate = self.universe.validator(COMPATIBILITY_PAIR_TABLE)
        duplicate.feed(records[0])
        with self.assertRaisesRegex(PairInventoryError, "strictly increasing"):
            duplicate.feed(records[0])

        unordered = self.universe.validator(COMPATIBILITY_PAIR_TABLE)
        unordered.feed(records[1])
        with self.assertRaisesRegex(PairInventoryError, "strictly increasing"):
            unordered.feed(records[0])

    def test_illegal_edge_and_complete_pair_id_are_rejected(self):
        record = next(self.universe.iter_expected_records(RESPONSE_PAIR_TABLE)).copy()
        record.update(
            source_world=0,
            target_world=3,
            pair_id="city:bank:alpha:0:3:1:0",
        )
        with self.assertRaisesRegex(PairInventoryError, "directed one-bit edge"):
            self.universe.validator(RESPONSE_PAIR_TABLE).feed(record)

        record = next(self.universe.iter_expected_records(RESPONSE_PAIR_TABLE)).copy()
        record["pair_id"] = record["pair_id"].removeprefix("city:")
        with self.assertRaisesRegex(PairInventoryError, "complete dataset identity"):
            self.universe.validator(RESPONSE_PAIR_TABLE).feed(record)

    def test_natural_incident_compatibility_edge_is_rejected(self):
        record = {
            "pair_id": "city:bank:alpha:1:3:1:3",
            "scene_index": 1,
            "bank_id": "city:bank:alpha",
            "source_world": 3,
            "target_world": 1,
            "position_index": 1,
        }
        with self.assertRaisesRegex(PairInventoryError, "natural-incident"):
            self.universe.validator(COMPATIBILITY_PAIR_TABLE).feed(record)

    def test_support_position_and_out_of_bounds_query_are_rejected(self):
        position = next(self.universe.iter_expected_records(RESPONSE_PAIR_TABLE)).copy()
        position.update(
            position_index=0,
            pair_id=(
                f"{position['bank_id']}:{position['source_world']}:"
                f"{position['target_world']}:0:{position['query_index']}"
            ),
        )
        with self.assertRaisesRegex(PairInventoryError, "evaluation positions"):
            self.universe.validator(RESPONSE_PAIR_TABLE).feed(position)

        query = next(self.universe.iter_expected_records(RESPONSE_PAIR_TABLE)).copy()
        query.update(
            query_index=12,
            pair_id=(
                f"{query['bank_id']}:{query['source_world']}:"
                f"{query['target_world']}:{query['position_index']}:12"
            ),
        )
        with self.assertRaisesRegex(PairInventoryError, "patch bounds"):
            self.universe.validator(RESPONSE_PAIR_TABLE).feed(query)

    def test_scene_bank_and_canonical_csv_integer_are_rejected_on_mismatch(self):
        base = next(self.universe.iter_expected_records(RESPONSE_PAIR_TABLE))
        mutations = (
            ({**base, "scene_index": "0"}, "scene"),
            ({**base, "bank_id": "city"}, "bank"),
            ({**base, "query_index": "00"}, "non-canonical"),
        )
        for row, message in mutations:
            with self.subTest(message=message):
                with self.assertRaisesRegex(PairInventoryError, message):
                    self.universe.validator(RESPONSE_PAIR_TABLE).feed(row)

    def test_dataset_edge_inventory_and_target_query_positions_fail_closed(self):
        class MissingEdgeDataset(_FakeDataset):
            def directed_edges(self, scene):
                yield from tuple(super().directed_edges(scene))[:-1]

        with self.assertRaisesRegex(PairInventoryError, "incomplete"):
            build_scene_pair_universe(MissingEdgeDataset(), 1)

        class NoTargetQueryDataset(_FakeDataset):
            position_roles = _FakeDataset.position_roles.copy()
            position_roles[1] = "support_pool"

        with self.assertRaisesRegex(PairInventoryError, "no query positions"):
            build_scene_pair_universe(NoTargetQueryDataset(), 1)


if __name__ == "__main__":
    unittest.main()
