#!/usr/bin/env python3
"""Compatibility shim: the real implementation now lives in
``petlab.ensemble``/``petlab.cli``. Kept so existing shell scripts calling
``python3 src/create_ensemble.py <config.json>`` keep working.
"""

import sys

from petlab.cli import main as petlab_main


def main(argv):
    """Ex: "python3 src/create_ensemble.py data/SPE1_Ensemble/SPE1_Poro.json" """
    petlab_main(["create-ensemble", *argv])


if __name__ == "__main__":
    main(sys.argv[1:])
