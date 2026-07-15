"""Eclipse-style deck templating and parsing.

The whole pipeline works by substituting ``$TOKEN`` placeholders inside a
plain-text ``.DATA``/``.INC`` deck with sampled values or control values, then
handing the result to a simulator binary (OPM ``flow`` or ``eclrun``). This
module holds that substitution logic (ported from ``src/create_ensemble.py``
and ``src/utils/utilities.py``) plus a minimal read-only deck parser (ported
from ``src/utils/deck_parser.py``) used to sniff units/keywords out of a deck.

Unlike the original ``replace_single_value``/``replace_random_field``/etc.,
these return the *sampled value* alongside the substituted text, not just the
text. That sampled value is what ``petlab.ensemble`` persists per realization
and what ``petlab.historymatch`` updates -- the previous code threw it away
after baking it into the deck as text, which made it impossible to run
ES-MDA against a scalar/array parameter after the fact.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gstools as gs
import numpy as np
import scipy.stats
from gstools import transform as gs_transform

from petlab.config import ParameterSpec, ControlSpec


# ---------------------------------------------------------------------------
# Parameter sampling (returns (deck_text, sampled_value))
# ---------------------------------------------------------------------------

def replace_single_value(d: dict) -> tuple[str, float]:
    p = d["parameters"]

    if d["name"] == "Normal":
        a = (p["min"] - p["mean"]) / p["std"]
        b = (p["max"] - p["mean"]) / p["std"]
        sample = scipy.stats.truncnorm.rvs(a, b)
        sample = sample * p["std"] + p["mean"]
    elif d["name"] == "LogNormal":
        a = (np.log(p["min"]) - np.log(p["mean"])) / np.log(p["std"])
        b = (np.log(p["max"]) - np.log(p["mean"])) / np.log(p["std"])
        sample = scipy.stats.truncnorm.rvs(a, b)
        sample = np.exp(sample * np.log(p["std"]) + np.log(p["mean"]))
    elif d["name"] == "Constant":
        sample = p["value"]
    else:
        raise ValueError(f"{d['name']} distribution not implemented yet")

    if p["type"] == "float":
        text = "%.5f" % sample
    elif p["type"] == "int":
        text = "%s" % int(sample)
    else:
        raise ValueError(f"Unknown parameter value type {p['type']!r}")

    return text, float(sample)


def replace_random_field(d: dict) -> tuple[str, np.ndarray]:
    p = d["parameters"]
    size = d["size"]
    x = np.arange(-0.5, size[0], 1)
    y = np.arange(-0.5, size[1], 1)
    z = np.arange(-0.5, size[2], 1)

    scale = d["scale"]
    angles = d["angles"]

    def _truncnorm_field(log_space: bool):
        model = gs.Gaussian(dim=3, var=1, len_scale=scale, angles=angles)
        srf = gs.SRF(model)
        srf((x, y, z), mesh_type="structured")
        if log_space == "arcsin":
            gs_transform.normal_to_arcsin(srf)
        fieldcdf = scipy.stats.norm.cdf(srf.field[:-1, :-1, :-1], 0, 1)

        if log_space:
            a = (np.log(p["min"]) - np.log(p["mean"])) / np.log(p["std"])
            b = (np.log(p["max"]) - np.log(p["mean"])) / np.log(p["std"])
            var = scipy.stats.truncnorm.ppf(fieldcdf, a, b)
            var = np.exp(var * np.log(p["std"]) + np.log(p["mean"]))
        else:
            a = (p["min"] - p["mean"]) / p["std"]
            b = (p["max"] - p["mean"]) / p["std"]
            var = scipy.stats.truncnorm.ppf(fieldcdf, a, b)
            var = var * p["std"] + p["mean"]
        return np.reshape(var, (size[0] * size[1] * size[2]), order="F")

    if d["name"] == "Normal":
        var = _truncnorm_field(log_space=False)
    elif d["name"] == "LogNormal":
        var = _truncnorm_field(log_space=True)
    elif d["name"] == "LogNormal-ArcSin":
        var = _truncnorm_field(log_space="arcsin")
    elif d["name"] == "Constant":
        var = np.full(size[0] * size[1] * size[2], p["value"], dtype=float)
    else:
        raise ValueError(f"{d['name']} distribution not implemented yet")

    if p["type"] == "float":
        text = "".join("%.5f " % v for v in var)
    elif p["type"] == "int":
        text = "".join("%s " % int(v) for v in var)
    else:
        raise ValueError(f"Unknown parameter value type {p['type']!r}")

    return text, var


def replace_incremental_text(d: dict, case_number: int) -> tuple[str, str]:
    p = d["parameters"]
    value = str(p["prefix"] + str(case_number) + p["suffix"])
    text = "'" + value + "'"
    return text, value


def replace_incremental_value(d: dict, case_number: int) -> tuple[str, np.ndarray]:
    """Load a previously-saved array (e.g. an ES-MDA posterior) from a
    ``.npy`` file named ``{prefix}{case_number}{suffix}`` and render it as a
    whitespace-separated deck array. This is the mechanism the CLRM loop uses
    to feed a posterior field back into the *next* generation of decks."""

    p = d["parameters"]
    file_path = str(p["prefix"] + str(case_number) + p["suffix"])
    row = np.load(file_path)
    text = "".join("%.5f " % float(v) for v in row)
    return text, row


def replace_incremental_array(d: dict, case_number: int) -> tuple[str, np.ndarray]:
    """Load one of several pre-existing plain Eclipse-format array includes
    (e.g. ``PERM1_ECL.INC``, which holds ``PERMX <values> / PERMY <values> /
    PERMZ <values> /`` back to back), named ``{prefix}{case_number}{suffix}``,
    extract the *first* ``KEYWORD <values> /`` block (dropping the keyword
    header and terminating ``/``, and ignoring any further blocks in the same
    file), and render the values as an inline deck array.

    This is the "pick one of N pre-generated full fields" mechanism (e.g. the
    Egg model's 100 permeability realizations) for the *first* generation of
    an ensemble. Unlike ``IncrementalText`` (which substitutes a filename for
    an ``INCLUDE``), this inlines the values directly so that later
    generations can swap in an ES-MDA posterior (``IncrementalValue``, a
    ``.npy`` array) at the exact same placeholder without changing the deck.
    """

    p = d["parameters"]
    file_path = str(p["prefix"] + str(case_number) + p["suffix"])
    with open(file_path, "r") as f:
        tokens = f.read().split()

    if tokens and tokens[0][0].isalpha():
        tokens = tokens[1:]  # drop the keyword, e.g. "PERMX"
    if "/" in tokens:
        tokens = tokens[: tokens.index("/")]  # keep only this one block, ignore any further ones

    values = np.array([float(t) for t in tokens])
    text = "".join("%.5f " % v for v in values)
    return text, values


def mutate_case(root_datafile_path: str, real_datafile_path: str, parameters: list[ParameterSpec], case_number: int) -> dict[str, Any]:
    """Substitute every ``ParameterSpec`` into the deck at
    ``root_datafile_path``, write the result to ``real_datafile_path``, and
    return the sampled value for each parameter (keyed by parameter name)."""

    with open(root_datafile_path, "r") as f:
        filedata = f.read()

    sampled: dict[str, Any] = {}
    for param in parameters:
        if param.type == "SingleValue":
            text, value = replace_single_value(param.distribution)
        elif param.type == "RandomField":
            text, value = replace_random_field(param.distribution)
        elif param.type == "IncrementalText":
            text, value = replace_incremental_text(param.distribution, case_number)
        elif param.type == "IncrementalValue":
            text, value = replace_incremental_value(param.distribution, case_number)
        elif param.type == "IncrementalArray":
            text, value = replace_incremental_array(param.distribution, case_number)
        else:
            raise NotImplementedError(f"Parameter type {param.type} is not recognized.")

        sampled[param.name] = value
        filedata = filedata.replace(param.name, text)

    Path(real_datafile_path).parent.mkdir(parents=True, exist_ok=True)
    with open(real_datafile_path, "w") as f:
        f.write(filedata)

    return sampled


# ---------------------------------------------------------------------------
# Control substitution
# ---------------------------------------------------------------------------

def apply_controls(base_datafile_path: str, real_datafile_path: str, controls: list[ControlSpec]) -> None:
    """Substitute every ``ControlSpec`` (well control) into the deck at
    ``base_datafile_path`` and write the result to ``real_datafile_path``."""

    with open(base_datafile_path, "r") as f:
        filedata = f.read()

    for control in controls:
        if control.kind == "float":
            text = "%.5f " % control.default
        elif control.kind == "int":
            text = "%s " % int(control.default)
        else:
            raise ValueError(f"Unknown control value type {control.kind!r}")
        filedata = filedata.replace(control.name, text)

    Path(real_datafile_path).parent.mkdir(parents=True, exist_ok=True)
    with open(real_datafile_path, "w") as f:
        f.write(filedata)


# ---------------------------------------------------------------------------
# Read-only deck parsing (unit/keyword sniffing)
# ---------------------------------------------------------------------------

class DeckParser:
    """Minimal Eclipse-deck reader: recursively follows ``INCLUDE`` and can
    search for a keyword's content anywhere in the deck (including includes).
    """

    def get_all_keywords(self, file_path: str, content: list | None = None) -> list:
        content = content if content is not None else []
        data_folder = os.path.dirname(file_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)

        with open(file_path, "r") as f:
            lines = f.readlines()

        is_include = False
        keyword = None
        _content: list = []

        for line in lines:
            l = line.rstrip().split()
            if len(l) == 0:
                continue
            if l[0][0] == "-":
                continue

            if l[0] == "INCLUDE":
                is_include = True
                if keyword is not None:
                    content.append((keyword, _content))
                keyword = l[0]
                _content = []
                continue

            if is_include:
                _content.append(l[0])
                content.append((keyword, _content))

                include_path = l[0][1:-1] if l[0][0] == "'" else l[0]
                file_path = os.path.join(data_folder, include_path)
                content = self.get_all_keywords(file_path, content)
                is_include = False
                continue

            if l[0].isalpha():
                if keyword is not None and keyword != "INCLUDE":
                    content.append((keyword, _content))
                keyword = l[0]
                _content = []
                continue

            _content.append(l)

        if keyword is not None:
            content.append((keyword, _content))

        return content

    def keyword_search(self, deck_path: str, keyword: str = "METRIC") -> tuple[bool, list]:
        """Search ``deck_path`` (following ``INCLUDE``s) for ``keyword`` and
        return ``(found, content_rows)``."""
        return self._read_file(deck_path, False, [], keyword)

    def _read_file(self, file_path: str, hit: bool, content: list, keyword: str) -> tuple[bool, list]:
        is_in = False
        is_include = False
        data_folder = os.path.dirname(file_path)

        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            l = line.rstrip().split()
            if len(l) == 0:
                continue
            if l[0][0] == "-":
                continue

            if l[0] == keyword:
                is_in = True
                hit = True
                continue

            if l[0] == "INCLUDE":
                is_include = True
                continue

            if is_include:
                include_path = l[0][1:-1] if l[0][0] == "'" else l[0]
                file_path = os.path.join(data_folder, include_path)
                hit, content = self._read_file(file_path, hit, content, keyword)
                is_include = False

            if is_in:
                if l[0].isalpha():
                    is_in = False
                    continue

                content.append(l)

                if keyword == "INCLUDE":
                    include_path = l[0][1:-1] if l[0][0] == "'" else l[0]
                    file_path = os.path.join(data_folder, include_path)
                    hit, content = self._read_file(file_path, hit, content, keyword)

        return hit, content
