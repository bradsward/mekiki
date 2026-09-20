# Action provenance: where did this dataset's actions come from?

Written before any code, like the other design docs here.

## The problem nobody audits

A robot-learning dataset's `action` column is supposed to be what the
operator (or policy) *commanded*. In practice many datasets are converted
from raw logs, and the conversion often **derives the action from the
recorded states** — `action[t] = state[t+1] - state[t]` — instead of storing
what was commanded. The result still looks like a normal action column. But:

- A policy trained on it learns to reproduce *hindsight state changes*, not
  operator intent. Controller lag, saturation, and contact all vanish from
  the label.
- Action-state consistency (M3) is **vacuous** on such a dimension: it holds
  by construction, so a zero residual carries no information.
- An off-by-one in that derivation (action labeled with the *previous*
  transition) is silent — nothing crashes, the policy just learns a lagged
  world.

Real example, found while validating M3: in `bridge_orig_lerobot` the
position and orientation action dimensions equal the recorded next-state
differences to 2e-9 (float32 precision) across 930 steps, while the gripper
action is a genuinely commanded target that the recorded gripper lags by
several frames. **One action vector, two different provenances.** Datasets
don't say which dimensions are which.

## What this audits

For each action dimension, from the data alone: *which recorded state
dimension does it most resemble, in what relation, at what lag — and is the
resemblance exact?*

Relations tested between action dimension `a` and state dimension `s`:

- **delta**: `a[t] ≈ s[t+1+L] - s[t+L]` — the action is a state *change*.
- **level**: `a[t] ≈ s[t+1+L]` — the action is an absolute *target*.

for a small set of lags `L` (default -2..+2). `L = 0` is the standard
convention (action at `t` causes the `t → t+1` transition).

A non-zero best lag alone is *not* called misalignment. A commanded action
that the robot follows through actuator lag also peaks at a positive lag;
that is dynamics. Only an **exact** match at a non-zero lag is flagged
(`misaligned`): an action equal to a shifted state difference to float
precision can't be explained by dynamics, only by an off-by-one in how it
was labeled. Non-exact best lags are still reported (with the full rho
profile), just not labeled a bug.

## Method

1. Pair actions with state changes/levels per episode. Pairs never cross an
   episode boundary.
2. For every (action dim, state dim, relation, lag), rank-correlate
   (Spearman). Rank correlation on purpose: robust to a few outliers (e.g.
   an angle wrapping through ±π) and to any monotone rescaling. Ties get
   average ranks (binary gripper actions are all ties), and rho is divided
   by the most those ties allow. A binary column against a continuous one
   tops out at sqrt(3)/2 = 0.866 even when the relationship is perfectly
   monotone, so an unadjusted rho would never call a commanded gripper
   "strongly related". The adjustment is the correlation of each column's
   tied ranks with the same ordering broken arbitrarily; it is 1 when
   there are no ties, so ordinary Spearman is unchanged.
3. The best candidate per action dimension is the largest `|rho|`.
4. For that candidate, fit `a ≈ k·y + b` by least squares with one round of
   outlier rejection (5 × MAD), and compute **`identity_fraction`**: the
   fraction of steps where `|a - y| <= identity_atol` — i.e. the action *is*
   the state quantity, not merely correlated with it.
5. Classify:

| classification | condition |
|---|---|
| `exact_state_difference` | relation delta, `identity_fraction >= 0.99` |
| `exact_state_level` | relation level, `identity_fraction >= 0.99` |
| `tracks_state_change` / `tracks_state_level` | `\|rho\| >= 0.9`, not exact |
| `weak_relationship` | `0.5 <= \|rho\| < 0.9` |
| `no_relationship_found` | `\|rho\| < 0.5` |

Also reported: the full `rho` at every lag for the winning
(state dim, relation) pair, so a lag conclusion can be checked rather than
trusted, and the margin over the runner-up so ambiguity is visible.

## Measured on real data

Run blind (no spec, no hint about which dimension is what), on local samples:

| dataset | dim | result |
|---|---|---|
| `bridge_orig_lerobot` (25 eps, 930 steps) | x, y, z, 3 orientation | `exact_state_difference`, lag 0, identity fraction 1.000, slope 1.00 |
| | gripper | `tracks_state_level` vs state 7, best lag +2, identity 0, rho 0.955 |
| `lerobot/pusht` (25 eps, 3122 steps) | x, y target | `tracks_state_level`, best lag +1, rho 0.999, identity 0 |
| `lerobot/xarm_lift_medium` (40 eps, 960 steps) | x, y, z | `tracks_state_change` / `weak_relationship`, lag 0, slope about 12 |

That is three different provenances. Bridge's pose actions are hindsight
state differences, so consistency checks on them are vacuous, while its
gripper is a real command the sensor follows two frames late. PushT's
actions are absolute cursor targets the agent follows about a frame behind.
xarm's actions are commanded deltas that the arm realizes at roughly a 12x
gain, so it correlates strongly with the state change but never equals it.
None of the three datasets says which of these it is.

The first cut of this had two real defects that the synthetic tests
caught before this table existed, both fixed:

- A binary gripper against a continuous state can't exceed rho of about
  0.866 even when perfectly monotone, so a real commanded gripper read as
  `weak_relationship`. Fixed with the tie adjustment above.
- A commanded action followed through actuator lag peaks at a non-zero
  lag, and I was calling that misalignment. Now only an exact match at a
  non-zero lag counts.

## What the relation label means when it isn't exact

Only an *exact* match separates "derived" from "commanded" with confidence.
For a non-exact result, `relation` just says which quantity the action
correlates with best, and for a commanded target it can be either. A robot
chasing a target moves toward it, so the state change (`delta`) tracks the
command about as well as the state level does; a binary gripper target
against a first-order actuator separates perfectly on the change and only
partly on the level. Don't read `tracks_state_change` on a gripper as "the
gripper action is a delta". The lag and `identity_fraction` are the parts
to trust.

## What this can and can't claim

- **It reports resemblance, never intent.** "Exact to 1e-5 across 930 steps"
  is overwhelming evidence the label was computed from the states; it cannot
  prove someone didn't *command* exactly that. The wording of every result
  says "consistent with derivation".
- **It proposes a correspondence; it doesn't adopt one.** This is the one
  place mekiki infers something from data — but the output is a *hypothesis
  with evidence* for a human to confirm and encode in an `ActionTargetSpec`,
  never something silently used. The project's rule (never guess semantics)
  holds; this tool is what makes a *good* declaration cheap to write.
- **Angle wrapping.** An Euler-difference action that wraps through ±π will
  mismatch a raw difference at the wrap. Rank correlation absorbs a few of
  these, but `identity_fraction` will read below 1 and the dimension will
  classify as `tracks_state_change`. Known, accepted, documented.
- **A sample, not the corpus.** Pairs are collected in memory, so the audit
  reads at most `max_episodes` episodes (default 200). Statistical, and the
  cap is the caller's to raise — nothing here materializes a full dataset.
- **Per-dimension only.** Cross-dimension structure (e.g. an action that is a
  *rotation* of a state difference — a frame mismatch) shows up as a weaker
  correlation, not a diagnosis. Frame diagnosis is M3's job, given a spec.
