import json
import io
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from formal_v2.formal_evaluation_streaming import (
    PROBE_TRAIN_BATCH_ROWS,
    _ProbeProgress,
    _ProgressHeartbeat,
    _available_probe_memory,
    _fit_probe_bundle,
    _fit_probe_bundle_exclusive,
    _probe_build_capacity,
)
from formal_v2.formal_evaluation import _prepare_alignment_shortcut_probes
from formal_v2.formal_probes import fit_action_response_probe, fit_select_compatibility_probe


class ProbeAccelerationTests(unittest.TestCase):
    def test_cli_exposes_probe_telemetry_without_changing_shard_progress(self):
        from formal_v2.formal_cli import main

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = root / "evaluation_state"
            state.mkdir()
            detail = _ProbeProgress(1, None).snapshot()
            (state / "probe_progress.json").write_text(json.dumps(detail))
            output = io.StringIO()
            with (
                mock.patch("formal_v2.formal_evaluation_resume.read_evaluation_status", return_value={"completed_units": 0}),
                mock.patch("sys.stdout", output),
            ):
                self.assertEqual(main(["evaluation-status", "--output", str(root)]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["completed_units"], 0)
            self.assertEqual(result["probe_progress"]["build_capacity"], 1)

    def test_memory_admission_retains_serial_fallback(self):
        gib = 1024**3
        self.assertEqual(_probe_build_capacity(2, None), 1)
        self.assertEqual(_probe_build_capacity(2, 300 * gib), 1)
        self.assertEqual(_probe_build_capacity(2, 575 * gib), 1)
        self.assertEqual(_probe_build_capacity(2, 576 * gib), 2)
        self.assertEqual(_probe_build_capacity(1, 1000 * gib), 1)

    def test_memory_probe_applies_cgroup_limit(self):
        gib = 1024**3
        files = {
            "/proc/meminfo": "MemAvailable: 734003200 kB\n",
            "/proc/self/cgroup": "0::/\n",
            "/sys/fs/cgroup/memory.max": str(400 * gib),
            "/sys/fs/cgroup/memory.current": str(120 * gib),
        }
        with (
            mock.patch.object(Path, "read_text", autospec=True, side_effect=lambda p: files[str(p)]),
            mock.patch.object(Path, "is_file", autospec=True, side_effect=lambda p: str(p) in files),
        ):
            self.assertEqual(_available_probe_memory(), 280 * gib)

    def test_two_admitted_builds_can_overlap(self):
        slots = threading.BoundedSemaphore(2)
        both_active = threading.Barrier(2)
        errors = []

        def fake_fit():
            both_active.wait(timeout=2)

        def run():
            try:
                _fit_probe_bundle_exclusive(slots)
            except BaseException as error:
                errors.append(error)

        with mock.patch("formal_v2.formal_evaluation_streaming._fit_probe_bundle", side_effect=fake_fit):
            threads = [threading.Thread(target=run) for _ in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(errors, [])

    def test_worker_telemetry_does_not_advance_authenticated_progress(self):
        detail = _ProbeProgress(2, 600 * 1024**3)
        unit = SimpleNamespace(execution_device="cuda:1", seed=42, arm="full", checkpoint_index=3)
        detail.callback(unit)({"probe": "response", "phase": "training", "step": 19, "total_steps": 2000})
        with tempfile.TemporaryDirectory() as temporary:
            store = SimpleNamespace(root=Path(temporary))
            with mock.patch("formal_v2.formal_evaluation_streaming._progress_update_if_reached") as update:
                heartbeat = _ProgressHeartbeat(
                    store, 0, seed=42, arm="full", substage="probe_state", shard="checkpoint-03",
                    probe_progress=detail,
                )
                heartbeat._publish()
                self.assertEqual(update.call_args.args[1], 0)
            record = json.loads((store.root / "probe_progress.json").read_text())
            self.assertEqual(record["workers"]["cuda:1"]["step"], 19)
            self.assertNotIn("sota_ready", record)
            self.assertNotIn("completed_units", record)

    def test_all_shortcut_probes_receive_gpu_and_batch_options(self):
        features = np.zeros((4, 2))
        rows = {"labels": np.array([0, 1, 0, 1])}
        for name in ("csi_only", "map_only", "scene_id_only", "edit_status_xor", "variant_id_match"):
            rows[f"{name}_shortcut_features"] = features
        lock = threading.Lock()
        with (
            mock.patch("formal_v2.formal_evaluation.fit_select_compatibility_probe", return_value=(object(), {"selected_family": "linear"})) as fit,
            mock.patch("formal_v2.formal_evaluation.predict_binary_probe", return_value=np.array([0.2, 0.8, 0.3, 0.7])) as predict,
        ):
            _prepare_alignment_shortcut_probes(
                42, rows, rows, {}, device="cuda:1", train_batch_rows=17, rng_lock=lock,
            )
        self.assertEqual(fit.call_count, 5)
        for call in fit.call_args_list:
            self.assertEqual(call.kwargs["device"], "cuda:1")
            self.assertEqual(call.kwargs["train_batch_rows"], 17)
            self.assertIs(call.kwargs["rng_lock"], lock)
        self.assertTrue(all(c.kwargs["batch_rows"] == 17 for c in predict.call_args_list))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_streaming_bundle_reaches_real_gpu_and_restores_cpu_payload_contract(self):
        rng = np.random.default_rng(381)
        x = rng.normal(size=(9, 4))
        rows = {"features": x, "labels": np.arange(9) % 2}
        response = {"features": x, "no_action_features": x * 0}
        for name in ("without_map", "edit_only", "csi_only", "oracle_x"):
            response[f"{name}_features"] = x
            response[f"{name}_zero_action_features"] = x * 0
        response.update(targets=x[:, :2], source_targets=x[:, :2] * 0)
        dataset = SimpleNamespace(indices_for_role=lambda role: np.array([0]))
        events = []
        with (
            mock.patch("formal_v2.formal_evaluation_streaming.route_dataset", return_value=object()),
            mock.patch("formal_v2.formal_evaluation_streaming.legacy._compatibility_dataset", return_value=rows),
            mock.patch("formal_v2.formal_evaluation_streaming.legacy._response_probe_dataset", return_value=response),
            mock.patch("formal_v2.formal_evaluation_streaming.legacy._prepare_alignment_shortcut_probes", return_value={}),
            mock.patch("formal_v2.formal_evaluation_streaming.fit_select_compatibility_probe", wraps=fit_select_compatibility_probe) as fit_binary,
            mock.patch("formal_v2.formal_evaluation_streaming.fit_action_response_probe", wraps=fit_action_response_probe) as fit_response,
        ):
            probes = _fit_probe_bundle(
                object(), dataset, object(),
                {"evaluation": {"probe_steps": 2, "probe_hidden_dim": 8, "probe_learning_rate": 0.001}},
                object(), object(), seed=42, batch_size=3, device="cuda:0",
                rng_lock=threading.Lock(), progress_callback=events.append,
            )
        self.assertEqual(fit_binary.call_count, 1)
        self.assertEqual(fit_response.call_count, 5)
        for call in (*fit_binary.call_args_list, *fit_response.call_args_list):
            self.assertEqual(call.kwargs["device"], "cuda:0")
            self.assertEqual(call.kwargs["train_batch_rows"], PROBE_TRAIN_BATCH_ROWS)
        self.assertEqual(next(probes.response_probe.parameters()).device.type, "cpu")
        self.assertTrue(any(event.get("step") == 2 for event in events))
        self.assertEqual(events[-1]["phase"], "ready_to_commit")
