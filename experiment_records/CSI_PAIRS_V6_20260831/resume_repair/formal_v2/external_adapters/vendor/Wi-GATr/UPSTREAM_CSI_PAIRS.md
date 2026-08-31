# Vendored Wi-GATr Snapshot

This directory is an unmodified source snapshot of the official Qualcomm AI Research
Wi-GATr repository, plus this provenance file.

- Paper: Hehn et al., "Differentiable and Learnable Wireless Simulation with Geometric Transformers"
- Paper artifact: `waibu/2406.14995v2.pdf`
- Upstream repository: `https://github.com/Qualcomm-AI-Research/Wi-GATr`
- Upstream revision recorded by the supplied GitHub archive: `6daa5bd49d9499817d903ab335830221d01e9daa`
- Supplied archive SHA-256: `b78be8ed14d16aab6c8fea54c2aa90a22bed492a12a6e892ee4c225c60a8138c`
- License: BSD-3-Clause-Clear; see `LICENSE`

CSI-PAIRS does not relabel this model as a native CSI predictor. The adapter keeps the
official geometric tokenizer, GATr regression backbone, scalar received-power objective,
and inverse-coordinate optimization. The adaptation is limited to deterministic conversion
of the frozen CSI-PAIRS 2.5D map into a triangular mesh, conversion of CSI to relative total
power in dB, source-role splitting, and the frozen six-condition evaluation protocol.

The original source requires Python 3.10. Its `uv.lock` and pinned GATr/WiInSim revisions
are retained. Scientific execution must use the dedicated environment created by
`formal_v2/external_adapters/setup_wigatr.sh`; it must not silently fall back to a substitute
network when an official dependency is unavailable.
