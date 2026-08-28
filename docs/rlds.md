# RLDS / Open X-Embodiment: format research notes

Checked the real on-disk format before writing any reader code, same as
`docs/episode.md` did for LeRobotDataset. What's here changes M1's plan —
see the recommendation in `STATE.md`. This is a reference doc, not a design
doc like `docs/episode.md`: it records what's actually there, not a
convention mekiki is choosing to enforce.

**Update:** the "LeRobotDataset isn't one format" finding below is no
longer just a finding — `src/mekiki/readers/lerobot.py` now handles both
the v3.0 and v2.0/v2.1 layouts. The RLDS-native (TFRecord/TFDS) parts of
this doc are still just notes; nothing's been built against them.

## What was checked

Open X-Embodiment's native release lives on Google Cloud Storage
(`gs://gresearch/robotics/<dataset>/<version>/`, also reachable over plain
HTTPS at `storage.googleapis.com/gresearch/robotics/...` — no GCS
credentials needed, it's a public bucket). Pulled `dataset_info.json` and
`features.json` for `dlr_sara_grid_clamp_converted_externally_to_rlds` (one
of the smaller OXE datasets) without downloading any of the actual episode
data, which is real TFRecord shards.

## The format is TFRecord-wrapped TFDS, not plain protobuf

`dataset_info.json` reports `"fileFormat": "tfrecord"` and
`"moduleName": "tensorflow_datasets.robotics.rtx.rtx"` — this is a real
`tensorflow_datasets` (TFDS) dataset, not a lightly-wrapped flat format.
`features.json` confirms the structure is genuinely nested: each episode is
a `FeaturesDict` containing a `steps` field, itself a TFDS *sequence*
feature (`Dataset` of per-step `FeaturesDict`s: `action`, `observation`
with `state`/`image`, `is_first`/`is_last`/`is_terminal`, `reward`,
`discount`, `language_instruction`), plus flat `episode_metadata`.

TFDS serializes nested/sequence features into TFRecord using its own
internal encoding rules (which sub-fields become flattened repeated
`tf.train.Feature` lists vs. nested serialized sub-`Example`s depends on
the TFDS version and feature types involved) — this is not a stable,
independently-documented wire format. Reading it correctly means either:

1. Using `tensorflow` / `tensorflow_datasets` itself to decode (heavy —
   TFDS pulls in TensorFlow, which is a large, partly GPU-oriented
   dependency in the same weight class as the torch dependency this
   project already explicitly refuses to make a core requirement).
2. Reverse-engineering TFDS's serialization scheme by hand and validating
   against real files without a reference decoder to check correctness
   against — a real risk of a *silent* decode bug, which is exactly the
   failure mode mekiki exists to catch in *other* people's data. Building
   one into mekiki's own reader would be a bad joke.

Neither is a "quick add." (1) is a real, deliberate dependency decision;
(2) isn't something to do without a reference implementation to check
against, which in practice means (1) anyway, just to validate.

## Images are individually PNG-encoded, not video

Unlike LeRobotDataset's `dtype: video` cameras (which need a real video
codec, e.g. av1), RLDS's `observation.image` here is
`encodingFormat: png` per frame — decodable with something as light as
Pillow, once the surrounding TFRecord/TFDS framing is actually parsed. So
the *pixel* half of this problem is easier than LeRobotDataset's; the
*framing* half is the hard part.

## Action semantics are documented in prose, not machine-readable

Better than LeRobotDataset in one respect: `features.json` carries a free-
text `description` per feature, e.g. this dataset's `action` is
`"[3x robot EEF position, 3x robot EEF orientation yaw/pitch/roll
calculated with scipy Rotation.as_euler(\"zxy\")]"`. That's genuinely more
than LeRobotDataset's `info.json` ever states. But it's prose, not a
structured field — still not something a reader can parse into an
`ActionDimSpec.mode`/`unit`/`frame` automatically, and this particular
description doesn't even say delta vs. absolute. The same fail-loudly rule
from `docs/episode.md` still applies; the description is worth surfacing to
whoever is writing the caller-supplied `ActionSpaceSpec` by hand, not worth
trying to parse.

## Open X-Embodiment is already reachable another way

Community conversions of most (all?) OXE datasets to LeRobotDataset format
already exist on the Hugging Face Hub — e.g. `IPEC-COMMUNITY/bridge_orig_lerobot`
is the exact Bridge V2 dataset this project's own README uses as its
running example, already in parquet+json form our LeRobotDataset reader
mostly knows how to read. Checked its `info.json`: `codebase_version:
"v2.0"`.

## LeRobotDataset itself isn't one format — v2.x is a different on-disk layout than v3.0

That last check surfaced something unrelated to RLDS but bigger: the
LeRobotDataset reader built this session only handles the v3.0 layout
(`meta/episodes/**.parquet` index + shared multi-episode data files, as
confirmed against `lerobot/pusht`). `IPEC-COMMUNITY/bridge_orig_lerobot`
(53,192 episodes) reports `codebase_version: "v2.0"`, and a couple of other
spot-checked community mirrors came back `v2.0`/`v2.1` too — only datasets
in the official `lerobot/` org's newer releases (checked:
`lerobot/aloha_sim_transfer_cube_human`) are on `v3.0`. LeRobotDataset v2.x
uses a different directory layout (one parquet file per episode, no
`meta/episodes/` index) that this reader does not understand yet — it would
currently either error or silently misread a v2.x dataset, neither of which
was tested. See the recommendation in `STATE.md`.
