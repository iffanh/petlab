#!/usr/bin/env python3
"""Deprecated: this script used to be the closed-loop workflow's only
automation -- manually copying the last optimized control from one
``study.json`` into a *different* config's parameters by hand, one stage at
a time. There is now a real closed-loop driver that does this automatically
(optimize -> apply to truth -> history-match -> repeat, with proper ES-MDA
updates in between, not just a control copy): see ``petlab clrm`` /
``petlab.clrm.run``, and ``docs/CLRM.md`` for how to set up a CLRM config.

This script's old logic depended on the previous ``study.json`` layout
(``study["extension"]["iterations"]["x"]``), which no longer exists now that
studies are built from :class:`petlab.ensemble.Study`, so it has not been
carried over.
"""

import sys


def main(argv):
    raise SystemExit(
        "src/apply_control.py has been replaced by the automated CLRM loop: "
        "run `petlab clrm <simulator> <clrm_config.json>` (see docs/CLRM.md)."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
