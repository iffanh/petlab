#!/usr/bin/env python3
"""Compatibility shim: the real implementation now lives in
``petlab.historymatch``/``petlab.cli``. Kept so existing shell scripts
calling ``python3 src/hm_ensemble.py <study.json> [method]`` keep working.

The default method is ESMDA (with PLSR dimensionality reduction bundled in,
see ``petlab.historymatch.esmda``); ``Spectral``/``PCESMDA`` are available as
optional backends. Unlike the old script, the number of PLSR components and
the polynomial order now come from the study's own
``historymatching.ncomponent``/``polynomial_order`` config fields rather
than extra positional CLI arguments -- edit the config JSON instead.
"""

import sys

from petlab.cli import main as petlab_main


def main(argv):
    """Ex: "python3 src/hm_ensemble.py simulations/studies/IE_PoroPerm2_RF.json ESMDA" """
    study_path = argv[0]
    cli_args = ["historymatch", study_path]
    if len(argv) > 1:
        cli_args += ["--method", argv[1]]
    if len(argv) > 3:
        print("Note: ncomponent/hyperparameter CLI overrides were removed; "
              "set historymatching.ncomponent / polynomial_order / alpha in the config JSON instead.")
    petlab_main(cli_args)


if __name__ == "__main__":
    main(sys.argv[1:])
