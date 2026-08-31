# CSI-PAIRS V6 executable data contract (V2.1)

Passing this schema is necessary, never sufficient, for scientific qualification. The runner rejects the previous V2.0 NPZ schema because it cannot encode V6 map, radio, split, provenance, or path semantics.

## Container and strict parsing

Use one compressed NumPy archive with `allow_pickle=False`. `metadata_json` and `engine_config_json` are parsed with duplicate-key and NaN/Infinity rejection. The canonical serialization of `engine_config_json` must hash to `metadata.engine.config_sha256`.

## Required arrays

| Array | Shape | Contract |
|---|---|---|
| `csi_repeat` | `[scene,world,position,repeat,channel]` | Independent real-then-imag observations |
| `csi_clean` | `[scene,world,position,channel]` | Pair-consistent clean target |
| `maps`, `noop_maps` | `[scene,world,map_channel,row,column]` | Canonical world and independent empty-edit rerender |
| `map_channel_names` | `[map_channel]` | Must include `occupancy`, `height`, `material` |
| `canonical_map_sha256`, `noop_map_sha256` | `[scene,world]` | Digest of every stored canonical rendering |
| `positions` | `[scene,position,2]` | BS-centered right-handed meter coordinates; within a city, the same physical coordinate must map to exactly one stable ID and role |
| `position_ids` | `[scene,position]` | Stable receiver-position identity used for city-level k; IDs and physical coordinates are one-to-one within each city |
| `free_space` | `[scene,world,position]` | Boolean common-free-space proof; every stored entry true |
| `radio_config` | `[scene,radio_feature]` | Carrier/array/antenna configuration supplied to F |
| `bs_pose` | `[scene,7]` | xyz plus scalar-first unit quaternion `(qw,qx,qy,qz)` in the frozen local frame (`wxyz` order) |
| `repeat_seeds` | `[scene,world,position,repeat]` | Unique observation-noise seeds across sibling worlds |
| `phase_reference_ids` | `[scene,position]` | Shared sibling-world phase/gauge reference identity |
| `phase_reference_values` | `[scene,position]` complex | Finite nonzero complex reference divided out of every sibling-world CSI value; a world axis is forbidden |
| `phase_reference_source_sha256` | `[scene,position]` | Lowercase SHA-256 of the exact calibration/renderer source record from which each complex reference was regenerated; a world axis is forbidden |
| `base_map_cluster_ids` | `[scene]` | Repeated foundations share split/city/statistical cluster |
| `world_bits` | `[world,bit]` | Complete binary hypercube |
| `primitive_ids`, `anchor_bits` | `[scene,bit]` | Per-bank primitive permutation and randomized anchor |
| `natural_world_index` | `[scene]` | Natural state row |
| `scene_ids`, `city_ids`, `bank_ids`, `scene_roles` | `[scene]` | Immutable identifiers and permission role |
| `position_roles` | `[scene,position]` | `standard` or target `support_pool/query` |
| `path_ids`, `path_power` | `[scene,world,position,path]` | Stable path IDs and nonnegative received powers |
| `path_surface_ids` | `[scene,world,position,path,interaction]` | Ordered/padded surface interactions |
| `noop_path_ids`, `noop_path_power` | `[scene,world,position,path]` | Canonical no-edit path retrace used to freeze `epsilon_path` |
| `noop_path_surface_ids` | `[scene,world,position,path,interaction]` | No-edit interaction retrace |
| `primitive_surface_ids` | `[scene,bit,surface]` | Surfaces affected by each registered primitive |

## Seven source roles

The exact source ledger is:

1. `source_encoder_train`
2. `source_method_selection`
3. `source_probe_train`
4. `source_probe_selection`
5. `source_calibration_fit`
6. `source_calibration_selection`
7. `source_final_unseen_bank`

`target` and `external_validation` are separate evaluation roles. Source and target city IDs must be disjoint. Formal configuration requires at least two source cities, two target cities, multiple independent banks per target city, and the configured bank minimum in every source role. A base-map cluster may not cross a role or city.

Every target city must contain at least `max(localization.label_budgets)` unique
`support_pool` physical positions. The runner checks this capacity before creating formal run
artifacts. Support/query exclusion compares both the stable ID and the BS-centered coordinate.

## Metadata

`metadata_json` has exact keys `schema_version`, `dataset_id`, `dataset_version`, `scientific_use`, `fixture`, `engine`, `representation`, `assets`, `generation`, and `external_reference`.

- Schema: `csi-pairs-formal-dataset-v2.1-v6`.
- `engine` includes exact `name`, `version`, `source_revision`, `license_id`, `config_sha256`, and `deterministic` fields.
- The current raw-complex P0 implementation requires
  `representation.phase_gauge_rule=shared_complex_reference`. The frozen V6
  `phase_invariant_delay_angle_power` fallback is not implemented and is rejected
  before training rather than silently changing the physical target. Each
  `(scene, position)` stores exactly one nonzero complex reference and its source-record
  SHA-256, with no world axis. Every sibling world applies
  `csi_gauge_fixed = csi_raw * conj(reference) / abs(reference)`.
- P0 sets `alignment_physical_representation=complex_csi_plus_delay_angle_power`; its source-train normalization is frozen and Response remains patch-local in the physical dead-zone units.
- `coordinate_system` is exactly `bs_centered_right_handed_meters`; map and position units are `m`.
- Antenna count, subcarrier count, `patch_antenna_size`, `patch_subcarrier_size`, and complex patch area must define an exact 2D tiling of the real-then-imag CSI grid.
- The resulting patch count must be a multiple of four. Stage-0 and the frozen model mask bank use
  an exact 75% hidden-patch cardinality; rounding to a nearby count is not admissible.
- `map_resolution_m` and `map_origin_xy_m` bind raster cells to the BS-centered meter frame used by path matching.
- `engine_config.path_identity` freezes ordered surface IDs, 1 ps delay bins,
  `theta_r/phi_r/theta_t/phi_t` in `1e-5` radian bins, and interaction vertices in
  `1e-5` meter bins. Sionna records with the same complete quantized physical
  identity are one numerical duplicate group whose path powers are summed; two
  different physical records may not share a path ID.
- Material values are categorical integers bounded by `assets.material_category_count`; model actions use explicit from/to planes, never category subtraction.
- `external_reference.available` must agree with actual external banks, but external validity remains `NOT_ASSESSED` until G8 evidence is run.
- Fixtures always use `fixture=true, scientific_use=FORBIDDEN`.

## Independent regeneration gate

The loader verifies shapes, finite values, hypercube completeness, canonical digests, engine-config digest, common-free-space declarations, phase/coordinate enums, categorical materials, unique repeat seeds, exact copied residual noise, role/city/bank constraints, path/no-op tensor consistency, and asset-field completeness.

For non-fixture data, every clean and no-op `(scene, world, position)` unit must
contain at least one registered RT path, and every clean CSI vector must have
nonzero norm. The loader rejects a candidate containing even one pathless or
all-zero clean unit before any teacher or model training starts.

Before qualification, `verify-data` executes a separately registered renderer command and compares regenerated maps, clean/repeated CSI, common-free-space masks, phase-reference IDs, complex reference values, reference-source SHA-256 values, paths, and no-op retraces against the archive independently for every scene. Engine source revision, engine license, asset licenses, dataset hash, and config hash are bound into its gate. Only `source_encoder_train` and `source_method_selection` failures are blocking for startup. Every later stage separately requires regeneration PASS for each role it reads, so a target/probe/calibration/final-unseen failure blocks that downstream stage without controlling startup. The fixture copy verifier is permanently non-scientific and refuses non-fixture input.

This regeneration gate still does not establish RT calibration or legal sufficiency. C11 requires independent RT calibration artifacts, and G8 requires an independent engine or controlled real intervention. Metadata `QUALIFIED` never supplies either result by itself.

When the primary dataset declares Sionna RT, a Sionna rerender from the same
engine family is not independent G8 evidence. Formal preflight rejects that
combination rather than allowing a same-simulator result to support C12.

## Evidence propagation

Every JSON, CSV row, checkpoint index, and manifest carries `dataset_sha256`, `config_sha256`, `fixture`, and `scientific_use`. JSON gates and manifests also bind the source-tree digest, requirements-lock digest, and structured runtime provenance. Qualification requires an authenticated regeneration gate produced by the same code and runtime. Downstream stages require matching hashes, the exact frozen teacher checkpoint hash, an authenticated V6 qualification gate, and per-role regeneration PASS. Risk archives additionally bind the executed checkpoint index and evaluation manifest. A non-fixture archive marked `CANDIDATE` cannot start factorial training; it must earn `FORMAL_EXPERIMENT_ALLOWED` from G1/G2.

G1 does not infer route noise tolerances from overall repeat NMSE. For each
`source_method_selection` bank, all unordered pairs of independent repeats at the same
scene/world/position are transformed with the source-encoder-train route normalization. The 0.95
quantile by default is computed separately for full-channel physical, full-channel teacher latent,
patch-local physical, and patch-local teacher latent RMS. The physical values govern the Alignment
and Response primary routes. Teacher latent values govern only independent sensitivity strata and
the auxiliary G2 teacher-alignment audit. Each configured null threshold must cover its corresponding
native-noise value or G1 fails. The rows and exact thresholds are bound in
`qualification/route_noise_floor.csv`; changing teacher latent values cannot change raw-CSI primary
sample inclusion.
