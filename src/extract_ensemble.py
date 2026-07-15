#!/usr/bin/env python3
"""Compatibility shim: the real implementation now lives in
``petlab.ensemble``/``petlab.cli``. Kept so existing shell scripts calling
``python3 src/extract_ensemble.py <study.json>`` keep working.
"""

import sys

from petlab.cli import main as petlab_main


def main(argv):
    """Ex: "python3 src/extract_ensemble.py simulations/studies/IE_Poro.json" """
    petlab_main(["extract", *argv])


if __name__ == "__main__":
    main(sys.argv[1:])
