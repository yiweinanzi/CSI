# Vendored PMNet Snapshot

This directory contains the unmodified PMNet v3 model source and MIT license from the
official `abman23/pmnet` repository, plus this provenance file.

- Papers: Lee et al., "PMNet: Robust Pathloss Map Prediction via Supervised Learning";
  Lee and Molisch, "A Scalable and Generalizable Pathloss Map Prediction"
- Upstream repository: `https://github.com/abman23/pmnet`
- Upstream revision: `a0e0c5926de721074beeb23f630f2d313f6508dd`
- Supplied archive: `waibu/PMNet-a0e0c592.zip`
- Supplied archive SHA-256: `e48e283e596270a9932052cc41f213b0a22c0ebbf29ba9551c5562d8cb877331`
- Vendored model-source SHA-256: `5ea7c5b4f0de14504c8e673a820a80452dd103f8c270a1b5d9533306d3d28c62`
- License: MIT; see `LICENSE`

CSI-PAIRS labels the integration `official-code-adaptation`, not a reproduction of the
upstream datasets or reported numbers. The adapter retains PMNet v3's ResNet encoder,
ASPP rates, multi-grid schedule, output stride, decoder, Adam optimizer, and 30-epoch
StepLR schedule. The first convolution is expanded so the formal occupancy, height,
material one-hot planes and BS transmitter raster can replace the upstream two image
channels. Training predicts total received-power radiomaps from source roles, with loss
only at registered receiver cells; six-condition localization never receives the true
receiver coordinate.
