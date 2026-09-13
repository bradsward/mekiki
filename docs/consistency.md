# Action-state consistency

This is the flagship check: does the recorded action actually explain the
observed state transition? It's what mekiki exists to do. Written before
any M3 code, per the roadmap, for the same reason `docs/episode.md` came
before any `Episode` code — the design decisions here are load-bearing for
everything downstream, and they're genuinely hard enough that getting them
wrong quietly would defeat the entire point of the tool. Don't skim this to
get to the code faster.

## What the check computes, precisely

For a single step `t → t+1`: given the proprioception at `t`, forward-
integrate the action recorded at `t` to produce a *predicted* proprioception
at `t+1`, then compare that prediction against the *actual* recorded
proprioception at `t+1`. The residual is the gap between them, in physical
units. Report the residual against an explicit tolerance — never a bare
pass/fail, same as every M2 check.

This is deliberately **single-step**, not a multi-step rollout. Forward-
simulating N steps ahead compounds integration error even for a perfectly-
behaved robot, since it's open-loop — nothing corrects the prediction with
real feedback between steps. That's a much harder, differently-scoped
problem (and arguably a different check entirely — "does this policy's
predicted trajectory track reality," not "is this one recorded action
consistent with this one recorded transition"). Out of scope here. Confusing
the two would produce residuals that grow with rollout length for reasons
that have nothing to do with data quality, which is exactly the kind of
false signal this tool is supposed to eliminate elsewhere.

## The correspondence problem

`docs/episode.md`'s `ActionDimSpec` says whether a dimension is absolute or
delta, its unit, and its frame. It deliberately does **not** say what
physical quantity a dimension predicts — that a `docs/episode.md` v1 reader
has no way to know, and guessing it would be exactly the kind of silent
guess this project refuses to make elsewhere (same principle as the
action-space fail-loud gate in `mekiki.readers.lerobot`).

But forward integration needs exactly that: to add a delta to *something*,
you have to know what that something is. `ActionDimSpec` alone can't answer
"does dimension 3 of this action vector move `ee_poses['ee'].position[0]`,
`ee_poses['ee']`'s orientation, `grippers['ee']`, or `joint_positions[2]`?"

So M3 introduces a new structure, **kept separate from `ActionDimSpec` and
`Episode` rather than extending them** — this is new complexity that only
M3 needs, and `docs/episode.md`'s abstraction is already shipped, tested,
and used by two reader layouts. Bolting M3-specific concerns onto it would
be a real design regression for a problem M3 can solve on its own:

```python
ActionTargetKind = Literal["position_axis", "orientation_axis", "gripper", "joint"]


@dataclass(frozen=True)
class ActionTarget:
    """What one action dimension predicts, physically.

    kind="position_axis" / "orientation_axis": end_effector names the key
        into Proprioception.ee_poses; axis is which of x/y/z (position) or
        which small-angle rotation-vector component (orientation) this
        dimension is. All three axes for a given (kind, end_effector) pair
        must be present together in the action space — see below.
    kind="gripper": end_effector names the key into Proprioception.grippers.
    kind="joint": index is the position into Proprioception.joint_positions.
    """

    kind: ActionTargetKind
    end_effector: str | None = None  # for position_axis / orientation_axis / gripper
    axis: Literal["x", "y", "z"] | None = None  # for position_axis / orientation_axis
    joint_index: int | None = None  # for joint


#: One entry per dimension in the episode's ActionSpaceSpec, same order,
#: same length. A dimension mekiki can't check yet (e.g. PushT's raw 2D
#: state — nothing in Proprioception structurally represents it) maps to
#: `None`, not a guess.
ActionTargetSpec = tuple[ActionTarget | None, ...]
```

`ActionTargetSpec` is caller-supplied, exactly like `ActionSpaceSpec` — it
comes from the same place (the dataset's documentation, card, or collection
code), never inferred from column names or shapes. A dimension mapped to
`None` is simply excluded from the check. An episode where every dimension
maps to `None` (PushT-shaped datasets: raw, uninterpreted `extra` state)
reports "0 dimensions modeled, nothing checked" — an honest degenerate
result, not an error and not a fabricated pass.

**Validation, done once per episode before any integration:** every
`position_axis` and `orientation_axis` triple (x, y, z for a given
`end_effector`) must be present together — a partial spec (only 2 of 3
axes) can't be integrated as a vector and is rejected loudly, not silently
integrated as a 2D quantity.

## Frame handling: what's actually supported

`docs/episode.md` is explicit that mekiki doesn't do general frame
conversion — readers preserve source frame labels, nothing more. M3 needs
*some* frame handling to do forward integration at all, so this section
draws the line precisely, rather than silently attempting something
unsound.

Two cases are supported, because both are fully computable from data
mekiki already has at hand — no external kinematics model, no URDF, no
robot-specific code required:

1. **The action's frame matches the target's own frame** (e.g. an action
   dimension declared `frame="base_link"` predicting a position axis whose
   `Pose.frame` is also `"base_link"`). Integration is direct: no rotation
   needed.
2. **The action's frame is `"ee"`** (end-effector-relative) while the
   target's own frame is something else (typically the base frame). This
   is resolvable using only the *current* (pre-action) orientation, which
   is already part of the recorded proprioception at `t` — no external
   model needed: rotate the delta vector into the target's frame using the
   rotation matrix built from `ee_poses[end_effector].orientation` at `t`,
   *then* integrate.

Any other frame mismatch — a named frame mekiki doesn't recognize as either
of the above, e.g. a sensor-mount frame requiring an extrinsic calibration
mekiki has no way to know — **fails loudly for that dimension** rather than
attempting an unsupported transform. This is the one place "fail loudly"
matters as much here as it does in `validate_action_space`: a silently wrong
frame transform would produce a residual that looks like a real
inconsistency (or worse, looks fine) for reasons that have nothing to do
with the data.

Position and orientation deltas are integrated **as whole 3-vectors**, never
per-axis independently — rotating one axis of a vector in isolation is
meaningless. This is the concrete reason position/orientation axes must be
validated as complete triples (previous section) before any integration
happens.

## Integration rules, per target kind

Let `dt = t_{k+1} - t_k` (already known non-negative and monotonic — M2's
`check_timestamp_monotonicity` is a precondition for this check the same
way it's a precondition for the M2 rate checks).

- **`position_axis`, mode `absolute`**: the 3 axes form an absolute target
  position, already in the target's own frame (frame mismatch here is
  rejected per above — an absolute target expressed in a *different* named
  frame than the state it predicts isn't one of the two supported cases,
  since there's no "current orientation to rotate by" story for an
  absolute value the way there is for a delta). Predicted position = the
  action's value, directly.
- **`position_axis`, mode `delta`**: predicted position = current position
  + delta (rotated into the target's frame first, if the action's frame is
  `"ee"` per above).
- **`orientation_axis`, mode `delta`**: the 3 axes form a rotation vector
  (axis = direction, magnitude = angle in radians — the standard
  small-angle/exponential-map parameterization). Predicted orientation =
  `q_current ⊗ q_delta`, where `q_delta` is that rotation vector's
  quaternion exponential. Quaternion composition, not axis-wise addition —
  orientation isn't Euclidean, and pretending it is would silently produce
  wrong residuals at exactly the poses where it matters most (rotations
  near ±180°, gimbal-adjacent configurations).
- **`orientation_axis`, mode `absolute`**: rejected for now, not
  implemented. An absolute 3-value orientation target is ambiguous without
  knowing its encoding (axis-angle? Euler, and in which order?) — that
  needs its own explicit, caller-declared field this doc isn't adding yet
  because no real dataset checked so far has needed it. Fail loudly rather
  than assume an encoding. Revisit when a real dataset actually needs this.
- **`gripper`**: predicted = the action's value, directly, **only if its
  unit is `"normalized"`** (matching `Proprioception.grippers`' own
  `[0, 1]` convention). Any other unit is rejected loudly rather than
  guessing a scale factor.
- **`joint`**: predicted `joint_positions[joint_index]` = current value +
  delta, or the action's value directly if absolute. No frame handling —
  joint angles aren't Cartesian, frame is meaningless for this kind and is
  ignored rather than checked.

## Residuals, in physical units

- **Position**: Euclidean distance between predicted and actual position,
  in meters, over the full 3-vector (never per-axis) — this is one
  physical displacement, not three independent numbers.
- **Orientation**: angular distance in radians:
  `θ = 2 · arccos(|dot(q_predicted, q_actual)|)`. The absolute value inside
  `arccos` is not optional — quaternions `q` and `-q` represent the same
  rotation, and skipping this turns every other correct prediction into a
  reported ~180° error.
- **Gripper**: `|predicted − actual|`, already normalized, reported as a
  fraction of full range.
- **Joint**: `|predicted − actual|` in radians, per joint, plus the max
  across joints as the episode-step's summary (a single joint badly wrong
  matters more than the average of many joints being fine).

## Tolerance: caller-supplied, never invented

A residual is only meaningful against a tolerance, and mekiki has no way to
know what counts as "normal" tracking error for an arbitrary robot — a
compliant arm in contact with an object will show real, physically genuine
deviation from a naive forward-integrated prediction that would be a real
red flag on a rigid, well-tuned industrial arm. Inventing a
one-size-fits-all default here would be exactly the kind of unvalidated
guess this project exists to catch in *other* tools' assumptions.

So tolerance is an explicit, caller-supplied model per target kind, with
two components:

```
tolerance(dt) = base_tolerance + rate_tolerance_per_second * dt
```

- `base_tolerance`: a floor independent of timestep — sensor/quantization
  noise, not integration error. In the residual's own units (meters,
  radians, normalized fraction).
- `rate_tolerance_per_second`: how much additional slack accumulates per
  second of integration window. This is the "scales with control
  frequency" part the roadmap calls out explicitly: a slower control loop
  (larger `dt` per step) gives the real robot more time to drift from a
  naive open-loop prediction before the next correction, so tolerance
  should grow with `dt`, not stay fixed. A linear model is the simplest
  defensible one and what this doc commits to for v1; both coefficients
  still come from the caller (derived from the dataset's own robot type
  and control loop, the same way `nominal_hz` does in M2), never invented
  by mekiki itself.

`dt` for a given step comes straight from the two frames' own timestamps —
already established as reliable by the M2 checks this one depends on.

## What v1 explicitly does not cover

- Multi-step rollout comparison (see above — a different, harder problem).
- Absolute-mode orientation targets (no declared encoding mechanism yet).
- Frame transforms beyond "same frame" and "current-orientation rotation
  from `ee`" — no general extrinsic/calibration-based transforms.
- Episodes where every action dimension maps to `None` in `ActionTargetSpec`
  (nothing modeled, nothing to report — not an error).
- Torque/force/wrench-based consistency (a genuinely different physical
  quantity from position/orientation kinematics; not attempted here).
