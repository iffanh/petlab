"""History-match backend registry.

``ESMDA`` (:func:`petlab.historymatch.esmda.update_plsr`) is the default --
see that module's docstring for why PLSR reduction is bundled into it rather
than being a separate optional step. ``Spectral`` and ``PCESMDA``
(:mod:`petlab.historymatch.extras`) are optional; each backend has a
different call signature (they need different inputs), so unlike
:mod:`petlab.optimization` this registry just exposes the callables by name
rather than a single dispatch function -- see ``petlab.cli``'s
``historymatch`` subcommand for how each one is actually invoked.
"""

from __future__ import annotations

from petlab.historymatch import esmda

DEFAULT_BACKEND = "ESMDA"


def get_backend(name: str):
    if name == "ESMDA":
        return esmda.update_plsr
    from petlab.historymatch import extras

    try:
        return extras.BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown history-match method {name!r}. Available: ESMDA, {', '.join(extras.BACKENDS)}") from None
