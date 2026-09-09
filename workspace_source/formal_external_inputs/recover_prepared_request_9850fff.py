from __future__ import annotations

import json
import secrets
from pathlib import Path

from formal_v2.formal_config import load_formal_config
from formal_v2.formal_dataset import FormalDataset
from formal_v2.formal_run_approval import preflight_full_run, write_approval_request


SOURCE = Path(
    "/root/xunlian/Futaoran/CSI_FACTORIAL_PILOT_FIX_20260825/"
    "code/CSI-PAIRS-v2.0-server"
)
OUTPUT = SOURCE / "runs/formal-v6-gpu-9850fff-20260825T071800Z"
CONFIG = SOURCE / "formal_v2/configs/formal_v2.json"
DATASET = Path(
    "/root/xunlian/Futaoran/正式开始训练/"
    "CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/dataset.npz"
)


def main() -> None:
    config = load_formal_config(CONFIG)
    dataset = FormalDataset.load(DATASET, require_clean_csi=True)
    input_values = {
        "adapter_manifest": str(
            SOURCE / "formal_v2/external_adapters/all_map_adapters_v1.json"
        ),
        "control_manifest": str(
            SOURCE / "formal_v2/external_adapters/resource_controls_v3.json"
        ),
        "scene_id_manifest": "builtin:sigmap-scene-id-v1",
        "shuffled_pair_manifest": str(
            SOURCE / "formal_v2/external_adapters/shuffled_pair_control_v3.json"
        ),
        "retention_manifest": str(
            SOURCE / "formal_v2/external_adapters/retention_control_v3.json"
        ),
        "representation_baseline_config": str(
            SOURCE / "formal_v2/configs/representation_baselines_v1.json"
        ),
    }
    preflight = preflight_full_run(
        config,
        dataset,
        OUTPUT,
        None,
        input_values,
        resource_registry=SOURCE / "formal_v2/configs/waibu_resources_v1.json",
        waibu_root=SOURCE / "waibu",
        resume=True,
    )
    result = write_approval_request(
        config,
        dataset,
        OUTPUT,
        preflight,
        run_nonce=secrets.token_hex(32),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
