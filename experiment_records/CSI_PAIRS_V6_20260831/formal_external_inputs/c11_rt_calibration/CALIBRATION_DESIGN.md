# C11 independent-engine calibration design

Status: thresholds pre-registered before generation or inspection of the held-out
Miami DiffeRT reference statistics.

## Engines and split

- Primary engine: Sionna RT 2.0.1, frozen project configuration and source assets.
- Independent reference engine: DiffeRT 0.10.0 at commit
  `673cc58ef61906b8ab0869dd206b3d032dbc01b2`, MIT license.
- Fit units: the 16 Denver `external_validation` base-map clusters.
- Validation units: the 16 Miami `external_validation` base-map clusters.
- One unit is one base-map cluster. Statistics are computed on the frozen natural
  world and aggregated across all 256 registered receiver positions.
- Fit and validation are disjoint by city, scene, base-map cluster, source asset,
  source record, and raw-unit identity.

## Frozen statistics

- `path_loss`: median receiver path loss in dB, calculated from the incoherent
  sum of valid per-path received powers for a 0 dBW transmitted signal.
- `delay_spread`: median receiver power-weighted RMS delay spread in ns.
- `angular_spread`: median receiver power-weighted circular RMS arrival-azimuth
  spread in degrees, using the DeepMIMO azimuth convention and the equivalent
  Sionna `phi_r` convention.
- `visible_path_count`: integer median number of valid paths within 30 dB of the
  strongest path at the receiver.

Per-receiver values use the mean across the two transmitter antenna elements.
No receiver, scene, path, or failed render may be excluded. A missing or nonfinite
statistic fails package generation.

## Frozen calibration model

For each statistic independently, fit an affine map with ordinary least squares
on the 16 Denver scene units. Apply it unchanged to the 16 Miami scene units.
Clamp the three nonnegative statistics to zero; round `visible_path_count` to the
nearest nonnegative integer. The adapter receives the independent-engine values
for fit units but receives only primary-engine statistics for validation units.
The outer formal runner, not the adapter, joins the held-out Miami reference.

## Frozen tolerances

- Path-loss MAE: 6 dB.
- Delay-spread MAE: 25 ns.
- Angular-spread MAE: 15 degrees.
- Visible-path-count MAE: 3 paths.

These thresholds were fixed at `2026-08-21T12:00:23Z`, before the held-out Miami
DiffeRT statistics were generated or inspected. They are not changed after a
failed result; failure requires claim downgrade or a separately approved new
protocol and fresh independent validation split.
