# C11 license and provenance verification

Verification date: 2026-08-21 UTC

Scope: mechanical verification of the license identifiers, local license files,
source revisions, and asset attribution used by the C11 independent-engine
simulation. This record is not legal advice and does not replace the external
human approval required for the formal run.

## Sionna RT primary simulation

- Sionna package version: `2.0.1`.
- Sionna RT package version: `1.2.1`.
- Frozen source revision: `04ddb9312116b408093b9d3ad363a3df355093a6`.
- Declared license: `Apache-2.0`.
- Local license file:
  `formal_v2/external_adapters/.runtime-sionna/src/sionna-main/LICENSE`.
- Local license SHA-256:
  `78d4571703b4ae47d96f32d678896a76749fbf0799db83c177f40be1c721375e`.
- The installed license text contains SPDX identifier `Apache-2.0`.

The C11 Sionna outputs are simulated ray-tracing statistics. They are not
measurements.

## DiffeRT independent reference simulation

- DiffeRT package version: `0.10.0`.
- Frozen source revision: `673cc58ef61906b8ab0869dd206b3d032dbc01b2`.
- Repository: `https://github.com/jeertmans/DiffeRT`.
- Declared license: `MIT`.
- Frozen provenance record:
  `formal_v2/external_adapters/DIFFERT_PROVENANCE.json`.
- Installed license file:
  `formal_v2/external_adapters/.runtime-differt/venv/lib/python3.12/site-packages/differt_core-0.10.0.dist-info/licenses/LICENSE.md`.
- Installed license SHA-256:
  `321b99fdc9938801c6610753b456a0c0d470f4a0337ea92655834aa8bb7697d2`.

The C11 DiffeRT outputs are simulated independent-engine reference statistics.
They are not measurements.

## Scene assets

- Source: OpenStreetMap Overpass building footprints.
- Declared source license: `ODbL-1.0`.
- Derived mesh and scene records are labeled `ODbL-1.0-DERIVED` in the asset
  manifest.
- Asset manifest:
  `/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/assets/asset_manifest.json`.
- Asset manifest SHA-256:
  `4547a46687201f2ab7d3dc92031e6e58dbf64a4b75fd796f6b946c228da479e7`.
- Attribution record:
  `/root/xunlian/Futaoran/正式开始训练/CSI-PAIRS-V6-WIDEBAND-FORMAL-227-20260819T105509Z/assets/ATTRIBUTION.md`.
- Attribution SHA-256:
  `18790a947887d57dd7ec10e961859adcc7977e994f92379d873750484cc1baba`.

## Scientific binding

- Formal dataset SHA-256:
  `060d8671380acaf43bd0beb2c77ac5c6ec112e4ddc083451ab36a5916b7c9cac`.
- C11 pre-registered protocol SHA-256:
  `040235346c65dcaca60c642b53ebef8a9a2c15c5fb6e2cb99b3d3358dbe5b96b`.
- Fit units: 16 independent Denver base-map clusters.
- Held-out validation units: 16 independent Miami base-map clusters.
- Statistic units: dB for path loss, ns for RMS delay spread, degrees for
  circular RMS arrival-azimuth spread, and integer path count for visible paths.
- Both engines use the frozen natural world, all 256 registered receiver
  positions per scene, the registered source assets, and the formal radio
  configuration. The engines remain distinct implementations and runtimes.

Any mismatch in these paths, hashes, versions, simulation labels, split roles,
or units invalidates this verification record and requires a fresh C11 package.
