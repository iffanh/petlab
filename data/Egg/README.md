# Egg Model

The Egg Model is an open-source synthetic reservoir benchmark (Jansen, J. D.,
Fonseca, R. M., Kahrobaei, S., Siraj, M. M., Van Essen, G. M., and Van den Hof,
P. M. J. (2014). *The egg model -- a geological ensemble for reservoir
simulation.* Geoscience Data Journal, 1(2), 192-195). It is a 60x60x7 channel
reservoir produced by 4 producers and 8 injectors, released together with an
ensemble of 100 equiprobable permeability realizations.

## What's in this folder

- `EggData.zip` -- the original release, as downloaded, kept as the
  source-of-truth artifact.
- `Egg_Model_Data_Files_v2/Eclipse/` -- the Eclipse-format subset extracted
  from the zip (`Egg_Model_ECL.DATA`, `ACTIVE.INC`, `COMPDAT.INC`,
  `mDARCY.INC`, `SCHEDULE_NEW.INC`). This is the *original* deck: constant
  well controls for the whole 10-year horizon, no `$PLACEHOLDER` tokens.
  Everything else in the zip (AD-GPRS/MRST/MoReS variants) is left
  compressed in `EggData.zip` since this codebase only drives Eclipse-format
  simulators (OPM `flow` / `eclrun`).
- `Egg_Model_Data_Files_v2/Eclipse/EGG_CLRM.DATA` -- a modified copy of the
  original deck, built for use with `petlab.clrm`:
  - `INCLUDE mDARCY.INC` (one fixed permeability field) is replaced with
    `PERMX\n$PERMFILE\n/` -- an inline array substituted by `petlab.deck`,
    either by picking one of the pre-generated realizations
    (`IncrementalArray`) or, for later CLRM stages, by an ES-MDA posterior
    field (`IncrementalValue`).
  - the single fixed `WCONPROD`/`WCONINJE` block is replaced by three
    checkpoints of 40x30-day `TSTEP`s each (same total 3600-day / ~9.86 year
    horizon and time discretization as the original schedule, just with
    control breakpoints inserted), with `$PRODnBHP{stage}` / `$INJECTnRATE{stage}`
    placeholders following the exact convention already used by
    `data/SPE10/spe10_layer21_ensemble/ECL_5SPOT_5C.DATA`. This is inlined
    directly into `EGG_CLRM.DATA` rather than a separate `INCLUDE`d file --
    `petlab.deck.apply_controls` only rewrites the one `.DATA` file it's
    given, it doesn't follow into includes, so every `$PLACEHOLDER` a control
    or parameter needs to reach must live in that one file (the original
    Egg deck's `ACTIVE.INC`/`COMPDAT.INC` are fine to keep as separate
    includes precisely because nothing ever needs to substitute into them).
  - `WBHP` was added to the `SUMMARY` section (the original deck only asked
    for `WOPR`/`WWPR`/`WWIR`/`WLPR`) so injector/producer bottom-hole pressure
    is available for history matching.
- `Egg_Model_Data_Files_v2/Permeability_Realizations/PERM{1..16,100}_ECL.INC`
  -- a curated 17-realization subset of the original 100 (realizations 1-16
  as the prior ensemble, realization 100 held out as the "truth" model), to
  keep the repo lean (~15 MB instead of ~84 MB for all 100). More
  realizations can be pulled back out of `EggData.zip` if needed:
  `Egg_Model_Data_Files_v2/Permeability_Realizations/PERM{i}_ECL.INC`.
- `Egg_CLRM_Prior.json` / `Egg_CLRM_Truth.json` -- the ensemble configs for
  the 16-member prior and the single held-out truth realization.
- `Egg_CLRM.json` -- the top-level `petlab.clrm` config tying the two
  together with the 3-stage schedule.

## Note on the previous configs

Earlier versions of this repo had `Egg.json`, `Egg_4Controls*.json`, etc.,
pointing at `./EggModel/data/...` -- an external folder excluded by
`.gitignore` and never actually present in this repo, so those configs could
not run. They have been removed and replaced by the configs above, which are
fully self-contained.
