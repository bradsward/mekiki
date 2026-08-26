# Episode abstraction

This document fixes the conventions the `Episode`/`Frame` types and every
reader (`LeRobotDataset`, RLDS/Open X-Embodiment, anything added later) must
follow. It exists because every downstream check — temporal integrity,
action-state consistency, coverage — inherits these choices. Getting them
wrong here means re-deriving them wrong in every check that touches poses or
actions. Written before any of that code, per the roadmap.

## Scope of one `Episode`

An `Episode` is one demonstration: a single continuous teleoperation attempt
at a task, from reset to termination, for one robot. It is not a dataset
(a directory of many episodes) and not a training batch (a resampled window
across episodes). A `Dataset`/corpus reader yields `Episode` objects lazily,
one at a time — an `Episode` itself is assumed small enough to reason about
in memory, but its camera streams are not eagerly decoded (see
[Streaming](#streaming-and-laziness)).

## `Frame`

One timestep within an episode:

| field | type | notes |
|---|---|---|
| `timestamp` | `float64`, seconds | relative to episode start (`t=0` at the first frame), **not** wall-clock. Monotonic non-decreasing is a property mekiki *checks* (M2) — the reader must not enforce or silently repair it. |
| `proprioception` | `Proprioception` | see below |
| `action` | `Action` | the commanded action recorded for this timestep — see [Action space](#action-space-delta-vs-absolute) |
| `images` | `dict[str, CameraFrame]` | keyed by camera name (e.g. `"wrist"`, `"exterior"`); each camera has its own timestamp, which may differ from `timestamp` — that gap is what M2's desync check measures, so it must survive the read |
| `is_first` / `is_last` | `bool` | episode-boundary flags, following RLDS naming since it already distinguishes these from a generic "done" |
| `success` | `bool \| None` | episode-level outcome label if the source dataset provides one; `None` if unlabeled, never coerced to `False` |
| `language_instruction` | `str \| None` | task description in natural language, if present |

`Proprioception` does not assume a single-arm-with-gripper robot. Real
datasets vary more than that: bimanual setups (e.g. ALOHA) have two
end-effectors and two grippers, many datasets record only joint positions
with no computed end-effector pose (that requires forward kinematics mekiki
doesn't perform), and some tasks aren't arm-shaped at all (e.g. PushT's raw
2D pusher position). So every field is optional except an escape hatch:

- `joint_positions` / `joint_velocities`: `float64[n_joints]`, radians and
  rad/s for revolute joints, or `None` if the dataset doesn't record them.
- `ee_poses: dict[str, Pose]`, keyed by a short name the source dataset uses
  to distinguish end-effectors (`"ee"` for a single arm, `"left"`/`"right"`
  for bimanual) — empty if the dataset provides no end-effector pose. Pose
  orientation is a unit quaternion `(x, y, z, w)` — never Euler angles,
  which are ambiguous without a fixed axis convention and lossy at
  gauge-lock — position in meters.
- `grippers: dict[str, float]`, each value normalized to `[0.0, 1.0]`
  (0 = fully closed, 1 = fully open) even when the source signal is binary,
  keyed the same way as `ee_poses` when there's a correspondence. Empty if
  there's no gripper.
- `extra: dict[str, NDArray[float64]]` for anything else the dataset
  records that doesn't fit the fields above, keyed by the source dataset's
  own field name, values as-recorded with **no assumed unit or frame** — a
  reader populating this must say so in its own docs, and mekiki does not
  infer physical meaning for it. This is a deliberate escape hatch, not a
  default: a reader should prefer the structured fields whenever a value
  clearly is a joint position, an end-effector pose, or a gripper.

## Action space: delta vs. absolute

This is the single most important convention in this document, because M3
(action-state consistency) forward-integrates recorded actions and compares
the result against recorded proprioception — that comparison is meaningless
if the integrator doesn't know whether an action is a delta or an absolute
target.

Datasets disagree, sometimes within a single episode across dimensions
(e.g. delta end-effector position, absolute gripper). mekiki does not infer
this at read time. Every `Episode` carries an explicit, per-dimension
`ActionSpaceSpec`:

```python
@dataclass(frozen=True)
class ActionDimSpec:
    name: str  # e.g. "x", "y", "z", "roll", "gripper"
    mode: Literal["absolute", "delta"]
    unit: Literal["m", "rad", "normalized"]
    frame: str  # coordinate frame this dimension is expressed in
```

`ActionSpaceSpec` is a tuple of `ActionDimSpec`, one per action vector
dimension, attached once per episode (or once per dataset, when a reader can
prove it's constant — LeRobotDataset states this in its feature schema;
RLDS/OXE state it per-dataset in the dataset card, not in the data itself).

**A reader that cannot determine the action space with confidence must fail
loudly rather than default to a guess.** A wrong default (e.g. assuming
absolute when the data is delta) silently corrupts every consistency check
built on top of it. This is the one place in mekiki where "fail loudly" beats
"report a magnitude" — there is no magnitude to report yet if the basic
semantics of the data are unknown.

## Coordinate frames

Every field that carries a pose or an action names its frame explicitly —
`frame: str` on `ActionDimSpec`, and a documented, fixed frame for
`Proprioception.ee_pose` (declared per-robot-embodiment, since it comes from
forward kinematics the reader doesn't recompute). Frame *names* are
free-text identifiers from the source dataset (`"base_link"`, `"panda_link0"`,
`"world"`, `"ee"`) — mekiki does not maintain a universal frame registry in
M1. It does not silently convert between frames at read time: converting a
delta action expressed in the end-effector frame into the base frame
requires the current end-effector orientation and is a kinematics operation,
which belongs to M3 (which already has to do forward integration), not to
the reader. **The reader's job is to preserve what the source dataset says,
labeled clearly enough that M3 can do frame conversion correctly** — not to
do the conversion itself.

## Camera streams

A `CameraFrame` is `{data: ArrayOrLazy, timestamp: float64, resolution:
tuple[int, int]}`. `data` is documented as HWC, uint8, RGB — readers convert
from whatever the source uses (LeRobotDataset: decoded video frames;
RLDS/OXE: raw or JPEG-encoded byte fields) so every downstream check sees one
consistent layout. Camera timestamps are independent of the frame's control
timestamp; if a source format doesn't record per-camera timestamps
separately (common — many datasets assume cameras are synced to the control
loop), the reader sets the camera timestamp equal to the frame timestamp and
records that assumption in the episode's metadata (`camera_timestamps:
"assumed synced"` vs. `"recorded"`), so M2's desync check knows whether an
absence of measured desync means "verified in sync" or "not measured."

## Streaming and laziness

Datasets are hundreds of gigabytes; no function may materialize a full
corpus in memory. Concretely:

- A dataset-level reader is a generator over `Episode` objects — one
  in flight at a time, never a list of all episodes.
- Within an `Episode`, proprioception and actions (small: a few floats per
  frame) are eagerly loaded — the whole per-frame numeric trajectory for one
  episode is negligible. Camera frames are the exception: `CameraFrame.data`
  is a lazy accessor (decode-on-access) backed by the source video/image
  reference, not a decoded array, so a check that never looks at pixels
  never pays for video decoding.

## What this document deliberately does not cover

- **Frame *conversion* math** (M3) — this doc only requires that source
  frames be labeled, not that mekiki can convert between them yet.
- **Resampling to a fixed control rate** — mekiki reports jitter and desync
  (M2) rather than hiding them by resampling; there is deliberately no
  "canonical frequency" concept here.
- **Per-dataset heuristics** for filling in an `ActionSpaceSpec` when a
  format's schema is genuinely ambiguous — that's reader-specific work for
  the LeRobotDataset and RLDS readers individually, not part of the shared
  abstraction.
