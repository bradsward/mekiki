# Idle time: how much of an episode is the robot just sitting there?

Written before the code, like the other design docs.

## What it's for

Demonstrations start before the operator moves and keep recording after the
task is done. Some also have mid-episode pauses. That time costs training
compute and, for a policy trained with behavior cloning, teaches "do nothing"
at the start state. The useful number per episode is **how much could be cut
without losing any motion**: leading dead time and trailing dwell. Interior
pauses are reported separately, because trimming those changes the trajectory
and might be the operator thinking, waiting on contact, or a gripper still
closing.

## What "idle" means here, and why it's declared

There's no universal speed below which a robot is idle. A 2 mm/s creep is
still while a peg is being inserted and dead time for a 20 cm/s pick-and-
place. So the thresholds are **caller-declared, per channel, in real units**,
never inferred from the data (same rule as `nominal_hz` and `ToleranceModel`).
Inferring "idle" from the episode's own speed distribution would call a slow
dataset idle everywhere and a fast one never.

A **motion channel** is one measured quantity plus the speed above which it
counts as moving:

| kind | speed | unit |
|---|---|---|
| `position` | end-effector position change | m/s |
| `orientation` | end-effector angular change (quaternion angular distance) | rad/s |
| `gripper` | change in normalized gripper state | 1/s |
| `joint` | one joint's change | rad/s (or m/s for prismatic) |
| `extra` | Euclidean norm change of a raw state array | caller's units per second |

An interval between two frames is **idle only if every declared channel is at
or below its threshold**. Requiring all channels matters: a gripper closing on
a stationary arm is not dead time. Declaring only `position` and getting
"idle" while the gripper works is the caller's declaration, and the result
lists which channels were used so that's visible.

## Segments

Consecutive idle intervals form a run; a run spans from the first frame of its
first interval to the last frame of its last interval. Runs shorter than
`min_idle_seconds` (caller-declared) are ignored, since every trajectory has
momentary stillness at direction changes. Each kept run is classified by
where it sits:

- `leading`: starts at the first frame. Pre-motion dead time.
- `trailing`: ends at the last frame. Post-task dwell.
- `interior`: neither. Hesitation.
- `whole_episode`: both. Nothing moved above the thresholds at all.

## What's reported

Per episode: every segment (kind, frame span, start time, duration), total
duration, `leading_seconds` / `trailing_seconds` / `interior_seconds`, the
`recoverable_fraction` (leading + trailing over total duration; this is the
part that can be trimmed without touching any motion), the thresholds and
`min_idle_seconds` used, and the **peak speed observed per channel** so a
threshold that is nonsense for this robot is visible (peak below threshold
means nothing ever counted as moving).

## Measured on real data

Position speed on `bridge_orig_lerobot` (25 episodes, 186 s, 0.2 s steps) and
the agent state on all 206 `lerobot/pusht` episodes (2544 s, 0.1 s steps):

| dataset | threshold | leading | trailing | interior |
|---|---|---|---|---|
| bridge | 0.01 m/s | 0.0% | 0.0% | 1.4% |
| bridge | 0.02 m/s | 0.0% | 0.0% | 8.8% |
| bridge | 0.03 m/s | 0.0% | 0.5% | 16.8% |
| pusht | 5 px/s | 0.0% | 0.0% | 0.0% |
| pusht | 20 px/s | 0.0% | 0.3% | 2.1% |
| pusht | 40 px/s | 0.0% | 2.2% | 10.8% |
| pusht | 80 px/s | 0.1% | 8.6% | 36.4% |

Two things this says. These public sets are already trimmed: there is
essentially no leading dead time at any threshold, and trailing dwell only
shows up once the threshold is loose. And the interior number is not a
property of the data alone, it moves from 0 to a third of the episode as
the threshold changes. That is why thresholds are declared and why every
result carries them and the peak speed: an idle percentage quoted without
its threshold means nothing. Median Bridge position speed is 6.6 cm/s and
only 1.4% of steps are under 5 mm/s, so a threshold picked from the
robot's speed scale (a few percent of typical speed) is a reasonable
starting point, but that choice stays with the caller.

The first, tighter thresholds I tried (5 mm/s position plus 0.02 rad/s
orientation plus a gripper channel) found nothing on Bridge. Measured
afterward: position alone is at or under 5 mm/s on 1.4% of steps, and
orientation is at or under 0.02 rad/s on only 1.2% (its median is 0.2
rad/s, the operator is rotating almost continuously), so requiring both
left 0.65% of steps and none of them in a run long enough to keep. Each
extra channel you require shrinks idle, which is the point of requiring
all of them, but a threshold set without looking at the channel's own
speed distribution can rule idle out entirely. Look at the distribution,
or at the peak speeds in the result, before declaring one.

## Limits

- Speeds come from finite differences of recorded state, so they inherit
  sensor noise. A threshold below the noise floor will never see idle; the
  peak-speed report is how to notice.
- It says nothing about *why* a pause happened, only that the declared
  channels were still.
- Timestamps must be strictly increasing (M2 checks that). A non-positive
  step raises instead of producing an infinite or negative speed.
- Recoverable time is a description, not a recommendation. Nothing here trims
  or filters; that is M9's job, and it decides from a policy, not from this.
