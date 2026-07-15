#!/usr/bin/env python3
"""Compatibility shim: the real implementation now lives in
``petlab.objective``/``petlab.cli``. Kept so existing shell scripts calling
``python3 src/evaluate_ensemble.py <simulator> <study.json> <controls.csv>``
keep working.
"""

import sys

from petlab.cli import main as petlab_main


def main(argv):
    petlab_main(["evaluate", *argv])


if __name__ == "__main__":
    main(sys.argv[1:])
