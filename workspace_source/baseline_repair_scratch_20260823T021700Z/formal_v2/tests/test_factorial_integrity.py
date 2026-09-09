from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from formal_v2.formal_config import load_formal_config
from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_factorial import (
    ARM_FACTORS,
    _build_corpus,
    _deterministic_adaptive_avg_pool2d,
    _identity_batch,
    _loss_components,
    _loss_weights,
    _make_plan,
    _measure_execution,
    _new_model,
    _response_batch,
    _normalized_action,
    _normalized_map,
    _training_normalization,
)
from formal_v2.formal_fixture import write_nonscientific_fixture
from formal_v2.formal_protocol import PatchSpec
from formal_v2.formal_protocol import typed_signed_edit
from formal_v2.formal_routing import fit_route_normalization
from formal_v2.formal_statistics import (
    exact_factorial_utilities,
    hierarchical_factorial_interval,
)
from formal_v2.formal_teacher import train_teacher_bundle


ROOT = Path(__file__).resolve().parents[2]
SMOKE_CONFIG = ROOT / "formal_v2" / "configs" / "formal_v2_smoke.json"


class ResponsePlanMutationTests(unittest.TestCase):
    @staticmethod
    def _corpus():
        entries = tuple(
            SimpleNamespace(query=query, mask=np.asarray([True, True], dtype=np.bool_))
            for query in (0, 0, 1, 1)
        )
        # Table values omit scene; _macro_sample prepends the table's scene.
        response_all = (
            [(0, target, 0, 0) for target in (1, 2, 3, 4)]
            + [(5, target, 0, 0) for target in (6, 7)]
        )
        routes = {
            (0, source, target, 0, 0): 2 if target in (1, 6) else 0
            for source, target, _position, _query in response_all
        }
        return SimpleNamespace(
            endpoint={0: [(0, 0)]},
            natural_endpoint={0: [(0, 0)]},
            alignment_active={0: [(0, 1, 0)]},
            alignment_null={0: [(0, 1, 0)]},
            response_all={0: response_all},
            response_active={0: [response_all[0], response_all[4]]},
            response_null={
                0: [response_all[1], response_all[2], response_all[3], response_all[5]]
            },
            response_bundles={
                0: [(0, (1, 2, 3, 4), 0, 0), (5, (6, 7), 0, 0)]
            },
            response_targets_by_source_query={
                (0, 0, 0, 0): (1, 2, 3, 4),
                (0, 5, 0, 0): (6, 7),
            },
            routed=SimpleNamespace(response_route=routes),
            dataset=SimpleNamespace(base_map_cluster_ids=np.asarray(["cluster"])),
            teacher=SimpleNamespace(mask_bank=entries),
        )

    def test_target_sampler_is_unconditional_and_not_fifty_fifty_reweighted(self):
        corpus = self._corpus()
        numerator = 0.0
        denominator = 0.0
        for step in range(200):
            plan = _make_plan(corpus, 64, 17, step)
            numerator += sum(
                weight * (unit[2] == 1)
                for unit, weight in zip(plan.response_all, plan.response_all_weights)
            )
            denominator += sum(plan.response_all_weights)
        # target=1 is one of six directed edges.  Source degrees are unequal,
        # so this also catches an unweighted expansion of complete bundles.
        self.assertAlmostEqual(numerator / denominator, 1.0 / 6.0, delta=0.02)

    def test_every_repeated_source_query_reuses_the_exact_mask_index(self):
        plan = _make_plan(self._corpus(), 128, 29, 4)
        observed = {}
        for units, masks in (
            (plan.response_all, plan.response_all_masks),
            (plan.response_active, plan.response_active_masks),
            (plan.response_null, plan.response_null_masks),
        ):
            for unit, mask in zip(units, masks):
                source_query = (unit[0], unit[1], unit[3], unit[4])
                observed.setdefault(source_query, set()).add(mask)
        self.assertTrue(observed)
        self.assertTrue(all(len(values) == 1 for values in observed.values()))
        self.assertAlmostEqual(sum(plan.response_all_weights), 128.0)
        self.assertAlmostEqual(sum(plan.response_active_weights), 128.0)
        self.assertAlmostEqual(sum(plan.response_null_weights), 128.0)
        all_targets = {}
        for unit in plan.response_all:
            all_targets.setdefault((unit[0], unit[1], unit[3], unit[4]), set()).add(
                unit[2]
            )
        self.assertEqual(all_targets[(0, 0, 0, 0)], {1, 2, 3, 4})
        self.assertEqual(all_targets[(0, 5, 0, 0)], {6, 7})
        self.assertEqual(
            plan,
            _make_plan(self._corpus(), 128, 29, 4),
            "the four arms must receive one deterministic common plan",
        )


class ArmExecutionMutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        path = write_nonscientific_fixture(Path(cls.temporary.name) / "fixture.npz")
        cls.dataset = FormalDataset.load(path)
        cls.config = load_formal_config(SMOKE_CONFIG)
        scenes = cls.dataset.indices_for_role("source_encoder_train")
        spec = PatchSpec.from_metadata(cls.dataset.metadata)
        teacher = train_teacher_bundle(
            cls.dataset.csi[scenes], spec, cls.config, seed=12031
        )
        route_normalization = fit_route_normalization(cls.dataset, teacher)
        normalization = _training_normalization(
            cls.dataset, scenes, route_normalization, spec
        )
        cls.corpus = _build_corpus(
            cls.dataset,
            scenes,
            teacher,
            cls.config,
            route_normalization,
            normalization,
        )
        cls.plan = _make_plan(cls.corpus, 4, 37, 0)

    def test_compact_corpus_sequences_enumerate_the_original_cartesian_contract(self):
        for scene in self.corpus.scenes:
            edges = tuple(self.dataset.directed_edges(scene))
            expected_endpoint = tuple(
                (world, position)
                for world in range(self.dataset.world_count)
                for position in range(self.dataset.position_count)
            )
            expected_response = tuple(
                (
                    edge.source_world,
                    edge.target_world,
                    position,
                    query,
                )
                for edge in edges
                for position in range(self.dataset.position_count)
                for query in range(self.corpus.teacher.patch_spec.patch_count)
            )
            self.assertEqual(tuple(self.corpus.endpoint[scene]), expected_endpoint)
            self.assertEqual(tuple(self.corpus.response_all[scene]), expected_response)
            expected_active = tuple(
                unit
                for unit in expected_response
                if self.corpus.routed.response_route[(scene, *unit)] == 2
            )
            expected_null = tuple(
                unit
                for unit in expected_response
                if self.corpus.routed.response_route[(scene, *unit)] == 0
            )
            self.assertEqual(tuple(self.corpus.response_active[scene]), expected_active)
            self.assertEqual(tuple(self.corpus.response_null[scene]), expected_null)
            for source in range(self.dataset.world_count):
                expected_targets = tuple(
                    sorted(
                        edge.target_world
                        for edge in edges
                        if edge.source_world == source
                    )
                )
                self.assertEqual(
                    self.corpus.response_targets_by_source_query[(scene, source, 0, 0)],
                    expected_targets,
                )
        self.assertFalse(
            any(isinstance(values, list) for values in self.corpus.response_all.values())
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_alignment_corpus_excludes_natural_incident_edges(self):
        for table in (self.corpus.alignment_active, self.corpus.alignment_null):
            for scene, rows in table.items():
                natural = int(self.dataset.natural_world_index[int(scene)])
                self.assertTrue(
                    all(
                        int(source) != natural and int(target) != natural
                        for source, target, _position in rows
                    )
                )

    def test_spatial_caches_match_independent_adaptive_pool_and_role_scope(self):
        expected_map_keys = {
            (int(scene), world)
            for scene in self.corpus.scenes
            for world in range(self.dataset.world_count)
        }
        expected_action_keys = {
            (int(scene), int(edge.source_world), int(edge.target_world))
            for scene in self.corpus.scenes
            for edge in self.dataset.directed_edges(int(scene))
        }
        self.assertEqual(set(self.corpus.normalized_maps), expected_map_keys)
        self.assertEqual(set(self.corpus.normalized_actions), expected_action_keys)

        def expected(values):
            tensor = torch.as_tensor(values, dtype=torch.float32)
            if tensor.shape[-2] >= 16 and tensor.shape[-1] >= 16:
                tensor = torch.nn.functional.adaptive_avg_pool2d(tensor, (16, 16))
            return tensor.numpy()

        map_key = next(iter(sorted(expected_map_keys)))
        direct_map = _normalized_map(
            self.corpus.normalization, self.dataset.maps[map_key]
        )
        np.testing.assert_allclose(
            self.corpus.normalized_maps[map_key],
            expected(direct_map),
            rtol=0.0,
            atol=1e-6,
        )
        action_key = next(iter(sorted(expected_action_keys)))
        scene, source, target = action_key
        direct_action = _normalized_action(
            self.corpus.normalization,
            typed_signed_edit(
                self.dataset.maps[scene, source],
                self.dataset.maps[scene, target],
                self.dataset.map_channel_names,
                self.corpus.material_categories,
            ),
        )
        np.testing.assert_allclose(
            self.corpus.normalized_actions[action_key],
            expected(direct_action),
            rtol=0.0,
            atol=1e-6,
        )
        self.assertFalse(self.corpus.normalized_maps[map_key].flags.writeable)
        self.assertFalse(self.corpus.normalized_actions[action_key].flags.writeable)

    def test_cached_map_and_action_batches_match_direct_model_preprocessing(self):
        unit = self.plan.response_all[0]
        entry = self.corpus.teacher.mask_bank[self.plan.response_all_masks[0]]
        scene, source, target, position, query = unit
        cached = _response_batch(self.corpus, [unit], [entry])
        model = _new_model(self.config, self.corpus, 20270815).eval()
        direct_map = torch.as_tensor(
            _normalized_map(
                self.corpus.normalization, self.dataset.maps[scene, source]
            )[None],
            dtype=torch.float32,
        )
        direct_action = torch.as_tensor(
            _normalized_action(
                self.corpus.normalization,
                typed_signed_edit(
                    self.dataset.maps[scene, source],
                    self.dataset.maps[scene, target],
                    self.dataset.map_channel_names,
                    self.corpus.material_categories,
                ),
            )[None],
            dtype=torch.float32,
        )
        with torch.no_grad():
            direct_state = model.state(
                cached["visible"], direct_map, cached["radio"], cached["masks"]
            )
            cached_state = model.state(
                cached["visible"], cached["maps"], cached["radio"], cached["masks"]
            )
            direct_outputs = model.predict(
                direct_state, direct_action, cached["query"]
            )
            cached_outputs = model.predict(
                cached_state, cached["action"], cached["query"]
            )
        torch.testing.assert_close(direct_state, cached_state, rtol=0.0, atol=1e-6)
        for direct, cached_output in zip(direct_outputs, cached_outputs, strict=True):
            torch.testing.assert_close(direct, cached_output, rtol=0.0, atol=1e-6)

    def test_numpy_pool_matches_torch_reference_for_nondivisible_grids(self):
        for shape in ((2, 3, 32, 32), (1, 2, 19, 23), (3, 17, 29)):
            values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            actual = _deterministic_adaptive_avg_pool2d(values, (16, 16))
            expected = torch.nn.functional.adaptive_avg_pool2d(
                torch.from_numpy(values), (16, 16)
            ).numpy()
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)

    def test_disabled_branches_are_not_forwarded_or_profiled_by_proxy(self):
        expected = {
            "endpoint": (2, 2),
            "alignment": (10, 10),
            "response": (5, 10),
            "full": (13, 18),
        }
        measured_flops = {}
        for arm, (alignment, response) in ARM_FACTORS.items():
            model = _new_model(self.config, self.corpus, 41)
            weights = _loss_weights(
                self.config, alignment, response, 1.0, 1.0, 1.0
            )
            execution = _measure_execution(
                model, self.corpus, self.plan, weights
            )
            self.assertEqual(
                (execution["state_calls"], execution["predict_calls"]),
                expected[arm],
            )
            self.assertEqual(
                execution["flop_measurement_status"],
                "TORCH_DISPATCH_COUNTER_PER_ARM",
            )
            measured_flops[arm] = execution["flops"]
            components = _loss_components(model, self.corpus, self.plan, weights)
            if alignment == 0:
                self.assertFalse(components["alignment"].requires_grad)
                self.assertEqual(float(components["alignment"]), 0.0)
            if response == 0:
                self.assertFalse(components["response"].requires_grad)
                self.assertEqual(float(components["response"]), 0.0)
        self.assertGreater(measured_flops["alignment"], measured_flops["endpoint"])
        self.assertGreater(measured_flops["response"], measured_flops["endpoint"])
        self.assertGreater(measured_flops["full"], measured_flops["alignment"])

    def test_response_pairing_break_keeps_action_but_replaces_target(self):
        first = self.plan.response_all[0]
        partner = next(
            unit
            for unit in self.plan.response_all
            if unit[-1] == first[-1]
            and (unit[0], unit[2], unit[3]) != (first[0], first[2], first[3])
        )
        entries = [
            self.corpus.teacher.mask_bank[self.plan.response_all_masks[0]]
        ]
        matched = _response_batch(self.corpus, [first], entries)
        shuffled = _response_batch(
            self.corpus, [first], entries, target_units=[partner]
        )
        self.assertTrue(torch.equal(matched["action"], shuffled["action"]))
        self.assertTrue(torch.equal(matched["source_z"], shuffled["source_z"]))
        self.assertFalse(torch.equal(matched["target_z"], shuffled["target_z"]))


def _statistics_rows():
    offsets = {"endpoint": 0.0, "alignment": 0.2, "response": 0.3, "full": 0.8}
    rows = []
    for city in ("city-a", "city-b"):
        for cluster_index in range(2):
            cluster = f"{city}-cluster-{cluster_index}"
            for bank_index, bank_value in enumerate((0.0, 1.0)):
                for seed in (1, 2):
                    for arm, offset in offsets.items():
                        rows.append(
                            {
                                "arm": arm,
                                "city_id": city,
                                "budget": 0,
                                "base_map_cluster_id": cluster,
                                "bank_id": f"{cluster}-bank-{bank_index}",
                                "canonical_bank_digest": f"digest-{cluster}-{bank_index}",
                                "seed": seed,
                                "draw": 0,
                                "utility_neg_log_median": bank_value + offset,
                            }
                        )
    return rows


class FactorialStatisticsMutationTests(unittest.TestCase):
    def test_selective_copied_bank_does_not_reweight_factorial_utility(self):
        rows = _statistics_rows()
        original = exact_factorial_utilities(rows, [0])
        copied = dict(rows[0])
        copied["bank_id"] = "renamed-copy"
        repeated = exact_factorial_utilities(rows + [copied], [0])
        self.assertEqual(original["utilities"], repeated["utilities"])
        self.assertEqual(original["interaction"], repeated["interaction"])

    def test_conflicting_copied_bank_is_rejected(self):
        rows = _statistics_rows()
        copied = dict(rows[0])
        copied["bank_id"] = "renamed-copy"
        copied["utility_neg_log_median"] = 1000.0
        forged = rows + [copied] * 100
        for statistic in (
            lambda: exact_factorial_utilities(forged, [0]),
            lambda: hierarchical_factorial_interval(forged, [0], 20, 9127),
        ):
            with self.assertRaisesRegex(
                ValueError, "conflicting utility values for canonical bank"
            ):
                statistic()

    def test_copied_bank_cannot_move_to_renamed_foundation(self):
        rows = _statistics_rows()
        source_foundation = rows[0]["base_map_cluster_id"]
        copied = [
            {**row, "base_map_cluster_id": "renamed-foundation"}
            for row in rows
            if row["base_map_cluster_id"] == source_foundation
        ]
        copied[0]["utility_neg_log_median"] = 1000.0
        forged = rows + copied
        for statistic in (
            lambda: exact_factorial_utilities(forged, [0]),
            lambda: hierarchical_factorial_interval(forged, [0], 20, 9127),
        ):
            with self.assertRaisesRegex(
                ValueError, "assigned to multiple independent units or cities"
            ):
                statistic()

    def test_primary_g4_intervals_are_one_synchronized_family(self):
        report = hierarchical_factorial_interval(
            _statistics_rows(), [0], 200, 9127
        )
        self.assertEqual(
            report["familywise_method"],
            "synchronized_bootstrap_max_absolute_deviation",
        )
        for name in ("interaction", "full_vs_alignment", "full_vs_response"):
            interval = report[name]
            self.assertLessEqual(
                interval["familywise_ci95_low"], interval["estimate"]
            )
            self.assertGreaterEqual(
                interval["familywise_ci95_high"], interval["estimate"]
            )

    def test_copied_bank_cannot_move_to_renamed_city(self):
        rows = _statistics_rows()
        source_foundation = rows[0]["base_map_cluster_id"]
        copied = [
            {**row, "city_id": "renamed-city"}
            for row in rows
            if row["base_map_cluster_id"] == source_foundation
        ]
        copied[0]["utility_neg_log_median"] = 1000.0
        forged = rows + copied
        for statistic in (
            lambda: exact_factorial_utilities(forged, [0]),
            lambda: hierarchical_factorial_interval(forged, [0], 20, 9127),
        ):
            with self.assertRaisesRegex(
                ValueError, "assigned to multiple independent units or cities"
            ):
                statistic()


if __name__ == "__main__":
    unittest.main()
