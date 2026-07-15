# Closed-Loop Reservoir Management (CLRM)

`petlab clrm` runs the receding-horizon loop described in the thesis (Ch. 7):
at each of a handful of checkpoints, re-optimize the remaining controls
against the current ensemble, apply the decided control to a held-out
"truth" model, history-match the ensemble against the truth's production
data so far, and repeat with the updated ensemble.

```
for each stage p:
    1. optimize  : DFTR over the current ensemble -> best control for the
                   remaining horizon; freeze this stage's slice of it
    2. apply     : run the (frozen-so-far + best-guessed-future) control on
                   the truth model
    3. history-match : truncate the truth's production data to this
                   checkpoint, ES-MDA (PLSR-reduced) the ensemble's log-PERMX
                   field against it
    4. regenerate: rebuild the ensemble's decks with the posterior PERMX
                   field, becoming the "current ensemble" for stage p+1
```

## Why re-simulate from t=0 every stage

Rather than an Eclipse `RESTART`-based warm start, every stage reruns every
realization (ensemble and truth) from the beginning with the full control
history applied so far. This is simpler and more robust -- no schedule/deck
surgery needed at runtime, no restart-file bookkeeping -- at the cost of
some redundant simulation. `data/Egg_CCS/WAG_BASE_RESTART` shows the
`RESTART` mechanism does exist and work in this codebase if a future version
of the loop wants to warm-start instead.

## Why PERMX, not the SPE10 layer-21 ensemble

An early version of this plan used `data/SPE10/spe10_layer21_ensemble`,
whose uncertain parameter is *which* of 16 pre-baked relative-permeability
curves each realization uses -- a discrete pick, not a continuous field
ES-MDA can update. The Egg model's uncertainty is the full permeability
field itself (100 open-source realizations), which is exactly what ES-MDA
is meant for, and matches the thesis's own real (non-toy) CLRM experiment.
See `data/Egg/README.md` for how the deck/data were adapted for this.

## Running the demo

You need an Eclipse-compatible simulator on your machine (OPM `flow` or
`eclrun`) -- there is no way around actually running the reservoir
simulations. From the repo root, with `petlab` installed
(`pip install -e .`, or just `pip install -r requirements.txt` and run the
`src/*.py` scripts / `python -m petlab.cli` directly):

```bash
petlab clrm /usr/bin/flow data/Egg/Egg_CLRM.json
```

This creates the 16-member prior ensemble and the single held-out truth
realization (Egg permeability realization #100), then runs the 3-stage loop
described above. Expect this to take a while on a real machine -- it's
16 (ensemble) x DFTR's internal iterations x 3 stages, plus 1 truth run per
stage, all real Eclipse-format simulations of a 60x60x7 model. Results
(per-stage diagnostics: predicted vs. truth objective, log-PERMX RMSE
before/after each update, the fully-applied control sequence) are written to
`simulations/clrm/egg_demo/clrm_result.json`.

To inspect one stage's intermediate state directly, look under
`simulations/clrm/egg_demo/stage_{n}/`: `ensemble/`/`truth/` hold the
simulated decks, `posterior/` holds that stage's ES-MDA posterior PERMX
arrays (`.npy`, one per realization), and `stage_{n+1}/` (built from the
`stage_n` posterior) holds the next generation's decks.

## What's verified without a simulator

This sandbox has no OPM `flow`/Eclipse install, so the real Egg run above
has not been executed end-to-end as part of building this. What *is*
verified (`tests/unit/`, no simulator needed): the ES-MDA (+ PLSR reduction)
math on a small synthetic problem, deck templating (including the
`$PERMFILE`/`$PRODnBHPk` substitutions `EGG_CLRM.DATA` needs), the
objective/constraint evaluator against fabricated `.npy` fixtures, and the
full stage-loop control flow (optimize -> apply -> truncate -> history-match
-> regenerate) against a stub simulator that mimics a reservoir's production
response with a simple analytic formula -- a test double, not a real
reservoir model, used only to prove the wiring is correct.

## Adapting this to a different model

A CLRM config (see `data/Egg/Egg_CLRM.json`) needs:

- `prior` / `truth`: two `EnsembleConfig` blocks (same shape as any other
  ensemble config in `data/`) sharing one deck, differing only in which
  realizations they cover and how many members they have.
- `truth_case_number`: which realization index is "truth" (held out from
  the prior).
- `stages`: a list of `{"checkpoint_years": ..., "control_indices": [...]}`
  -- `control_indices` are positions into the shared `controls` list that
  get frozen once that stage's optimization finishes.
- `updated_parameter`: the name of the `ParameterSpec` that ES-MDA updates
  each stage (must be the one whose `Distribution` picks a full field --
  `IncrementalArray` for the first generation, switched to
  `IncrementalValue` automatically for later generations).

The deck itself needs: (1) that parameter's placeholder positioned so its
substituted value is a plain inline array (e.g. `PERMX\n$PERMFILE\n/`, *not*
inside an `INCLUDE`, since later generations substitute raw numbers, not a
filename), and (2) `WCONPROD`/`WCONINJE` blocks with per-stage
`$CONTROLNAMEk` placeholders between `TSTEP` groups, following the exact
convention already used by `data/SPE10/spe10_layer21_ensemble/ECL_5SPOT_5C.DATA`.
