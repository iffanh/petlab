"""Deck templating tests -- no simulator needed, just string substitution
and file I/O."""

import numpy as np
import pytest

from petlab import deck
from petlab.config import ControlSpec, ParameterSpec


def test_replace_single_value_constant():
    text, value = deck.replace_single_value({"name": "Constant", "parameters": {"value": 3.5, "type": "float"}})
    assert value == 3.5
    assert text == "3.50000"


def test_replace_single_value_normal_is_bounded():
    d = {"name": "Normal", "parameters": {"mean": 10.0, "std": 1.0, "min": 8.0, "max": 12.0, "type": "float"}}
    for _ in range(20):
        text, value = deck.replace_single_value(d)
        assert 8.0 <= value <= 12.0
        assert text == f"{value:.5f}"


def test_replace_incremental_array_strips_keyword_and_slash(tmp_path):
    inc_path = tmp_path / "PERM1_ECL.INC"
    inc_path.write_text("PERMX\n1.0 2.0 3.0\n4.0\n/\n")

    d = {"parameters": {"prefix": str(tmp_path / "PERM"), "suffix": "_ECL.INC"}}
    text, values = deck.replace_incremental_array(d, 1)

    np.testing.assert_allclose(values, [1.0, 2.0, 3.0, 4.0])


def test_replace_incremental_array_ignores_later_blocks_in_same_file(tmp_path):
    """The real Egg PERM*_ECL.INC files hold PERMX/PERMY/PERMZ back to back
    in one file; only the first block should be picked up."""
    inc_path = tmp_path / "PERM1_ECL.INC"
    inc_path.write_text("PERMX\n1.0 2.0\n/\nPERMY\n30.0 40.0\n/\nPERMZ\n5.0 6.0\n/\n")

    d = {"parameters": {"prefix": str(tmp_path / "PERM"), "suffix": "_ECL.INC"}}
    text, values = deck.replace_incremental_array(d, 1)

    np.testing.assert_allclose(values, [1.0, 2.0])
    assert text.split() == ["1.00000", "2.00000"]


def test_replace_incremental_value_reads_npy(tmp_path):
    arr = np.array([1.1, 2.2, 3.3])
    np.save(tmp_path / "PERMX_7.npy", arr)

    d = {"parameters": {"prefix": str(tmp_path / "PERMX_"), "suffix": ".npy"}}
    text, values = deck.replace_incremental_value(d, 7)

    np.testing.assert_allclose(values, arr)


def test_mutate_case_substitutes_every_parameter(tmp_path):
    root = tmp_path / "ROOT.DATA"
    root.write_text("PORO\n$POROVAL\n/\nPERMX\n$PERMFILE\n/\n")

    perm_inc = tmp_path / "PERM3_ECL.INC"
    perm_inc.write_text("PERMX\n100 200 300\n/\n")

    params = [
        ParameterSpec(name="$POROVAL", type="SingleValue", distribution={"name": "Constant", "parameters": {"value": 0.2, "type": "float"}}),
        ParameterSpec(name="$PERMFILE", type="IncrementalArray", distribution={"parameters": {"prefix": str(tmp_path / "PERM"), "suffix": "_ECL.INC"}}),
    ]

    out_path = tmp_path / "case_3" / "ROOT_3.DATA"
    sampled = deck.mutate_case(str(root), str(out_path), params, case_number=3)

    assert sampled["$POROVAL"] == 0.2
    np.testing.assert_allclose(sampled["$PERMFILE"], [100, 200, 300])

    written = out_path.read_text()
    assert "$POROVAL" not in written
    assert "$PERMFILE" not in written
    assert "0.20000" in written
    assert "100.00000" in written


def test_apply_controls_formats_by_kind(tmp_path):
    base = tmp_path / "BASE.DATA"
    base.write_text("WCONPROD\n$PROD1BHP1 $STEPS\n/\n")

    controls = [
        ControlSpec(name="$PROD1BHP1", default=395.123, kind="float"),
        ControlSpec(name="$STEPS", default=30.0, kind="int"),
    ]

    out_path = tmp_path / "OUT.DATA"
    deck.apply_controls(str(base), str(out_path), controls)

    written = out_path.read_text()
    assert "395.12300" in written
    assert "30 " in written
