#!/usr/bin/env python3
"""Compatibility shim: the real implementation now lives in
``petlab.objective``/``petlab.optimization``/``petlab.cli``. Kept so
existing shell scripts calling
``python3 src/optimize_ensemble.py <simulator> <study.json>`` keep working.

The default optimizer is DFTR (``py_trsqp``); COBYLA/COBYQA/NOMAD/BO/STOSAG
are available as optional backends -- see ``petlab.optimization.extras``.
"""

import sys

from petlab.cli import main as petlab_main


def main(argv):
    """Ex: "python3 src/optimize_ensemble.py /usr/bin/flow simulations/studies/IE_PoroPerm_Opt_RandomField.json" """
    petlab_main(["optimize", *argv])


if __name__ == "__main__":
    main(sys.argv[1:])
