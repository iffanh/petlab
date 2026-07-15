# petlab

## An open-source framework for Closed-Loop Reservoir Management (CLRM).

`petlab` mutates Eclipse-style `.DATA` decks by substituting `$PLACEHOLDER`
tokens, runs them through a simulator binary (OPM `flow` or `eclrun`), and
reads the results back via `resdata` -- so it works non-intrusively with any
Eclipse-compatible simulator. The actual implementation lives in
`src/petlab/`, installable as a package (`pip install -e .`) with a single
CLI entrypoint, `petlab`:

```
petlab create-ensemble <config.json>       # sample uncertain parameters, write one deck per realization
petlab run <simulator> <study.json>        # apply the configured default controls and simulate
petlab extract <study.json>                # pull summary/3D vectors out as .npy files
petlab optimize <simulator> <study.json>   # optimize controls (default: DFTR)
petlab historymatch <study.json>           # update parameters against observed data (default: ES-MDA)
petlab evaluate <simulator> <study.json> <controls.csv>   # evaluate one specific control vector
petlab clrm <simulator> <clrm_config.json> # the full closed loop: optimize -> apply to truth -> history-match -> repeat
```

The old `src/create_ensemble.py`/`run_ensemble.py`/etc. scripts still work
(they're now thin shims calling into `petlab.cli`), so existing shell
scripts under `examples/` don't need to change.

- **History matching**: Ensemble Smoother with Multiple Data Assimilation
  (ES-MDA, with PLSR dimensionality reduction bundled in for full-field
  parameters) is the default. A standalone PLSR + Polynomial Chaos
  Expansion "Spectral" method (no ES-MDA) and a PCE-surrogate ES-MDA variant
  are available as optional, non-default methods --
  see `petlab.historymatch.extras`.
- **Optimization**: the derivative-free trust-region filter method ("DFTR",
  `py_trsqp`) is the default. COBYLA, COBYQA, NOMAD, Bayesian Optimization
  and StoSAG are available as optional backends (each needs its own extra
  package, see `pyproject.toml`'s `[project.optional-dependencies]`) --
  see `petlab.optimization.extras`.
- **Closed-loop management**: `petlab clrm` runs the actual receding-horizon
  MPC loop (thesis Ch. 7) -- see `docs/CLRM.md` and the Egg model demo under
  `data/Egg/`.

## Configuring a study



<details><summary> <b> flash_calculation.py </b> </summary>

Calculate liquid and gas composition (flash calculation) given fluid composition, pressure and temperature condition. (Now it works only on field unit)

Run flash_calculation.py with the following command:

```python flash_calculation.py /path/to/fluid_dataset.json/ /path/to/result.csv/```

Sample data set is given under the json_files folder. Run the sample data as follows:

``` python flash_calculation.py ./json_files/flash_calculation/pvt_dataset_2.json ./results/pvt_dataset_2_result.csv ```

The .JSON file must include the following:
``` 
  "PressureK" : Convergence pressure (psi) 
  "Pressure" : Pressure condition for flash calculation (psi)
  "Temperature" : Temperature condition for flash calculation (Rankine)
  "A0" : A0 variable
  "fv" : Initial condition for fv
  "max_iter" : Maximum number of iteration to get fv
  "Component" : {
    "<Component Name 1>" : {
      "Mole_Fraction" : total component fraction
      "Critical_Pressure" : critical pressure of the component (psia)
      "Critical_Temperature" : critical temperature of the component (psia) 
      "Accentric_Factor" : accentric factor of the component
      }
    }
```

</details>

## Installation

```shell
pip install -e .                 # core package + petlab CLI (numpy/scipy/resdata/gstools/scikit-learn/casadi/...)
pip install -e ".[dftr]"         # + the default optimizer backend (py_trsqp)
pip install -e ".[spectral]"     # + chaospy, for the optional Spectral/PCESMDA history-match methods
pip install -e ".[dev]"          # + pytest, to run tests/unit (no simulator needed)
```

You also need an Eclipse-compatible simulator on your `PATH` (OPM `flow` or
`eclrun`) to actually run anything beyond `create-ensemble` -- there's no way
around that, `petlab` only orchestrates the simulator, it doesn't replace
it. `requirements.txt` is kept for the old `pip install -r requirements.txt`
workflow, but the `pyproject.toml` extras above are the more precise/lighter
option (e.g. it doesn't force a C-compiler-requiring `chaospy` install on
you just to use the default ES-MDA + DFTR path).