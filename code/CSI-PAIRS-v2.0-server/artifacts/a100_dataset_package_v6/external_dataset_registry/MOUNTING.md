# External dataset use and mount policy

The public external data is physically included in this delivery for A100-side
baseline work. Inclusion does not make it part of the CSI-PAIRS paired-world
dataset.

Keep these roots separate in experiment configuration:

```text
CSI_PAIRS_MAIN=/path/to/newly-generated-and-qualified/dataset.npz
DEEP_MIMO_ROOT=/path/to/external_public_datasets/external_wireless/DeepMIMO
URBAN_MIMO_ROOT=/path/to/external_public_datasets/external_wireless/UrbanMIMOMap
RADIO_MAP_ROOT=/path/to/external_public_datasets/external_wireless/RadioMapSeer
DEEP_SENSE_ROOT=/path/to/external_public_datasets/external_wireless/WWM/alternatives/DeepSense6G
```

An external dataset may be consumed only by a named baseline, adapter, or
preregistered external-domain experiment. Do not point CSI-PAIRS Stage-0,
route selection, normalization selection, or the formal sibling-world loader
at any of these roots implicitly.

Large upstream ZIP files are preserved as downloaded. The duplicate extracted
`RadioMapSeer/data/` tree is omitted to avoid adding roughly 3.9 GB and more
than 355,000 redundant files; extract `RadioMapSeer.zip` on the server when a
baseline needs those files. The DeepMIMO virtual environment and Git object
databases are also omitted because Linux must build its own reviewed runtime.
