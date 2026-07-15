"""Config (de)serialization round-trip tests.

``EnsembleConfig`` and ``OptimizationConfig`` both use the on-disk key
"parameters" for two different things (the ensemble's uncertain
``ParameterSpec`` list vs. the optimizer's settings dict, per the legacy
``{"optimization": {"parameters": {...}}}`` schema) -- ``to_dict`` must keep
``optimization`` nested rather than spread at the top level, or the two
"parameters" keys collide and one silently clobbers the other.
"""

from petlab.config import EnsembleConfig, ParameterSpec, ControlSpec, OptimizationConfig


def test_ensemble_config_round_trip_keeps_parameters_and_optimization_distinct():
    config = EnsembleConfig(
        name="test",
        ne=2,
        root="dummy.DATA",
        parameters=[ParameterSpec(name="$PERMFILE", type="IncrementalArray", distribution={"parameters": {"prefix": "p", "suffix": "s"}})],
        controls=[ControlSpec(name="$C1", default=1.0, lb=0.0, ub=2.0)],
        optimization=OptimizationConfig(optimizer="DFTR", cost_function="NPV", max_iter=42),
    )

    d = config.to_dict()
    assert d["parameters"] == [{"Name": "$PERMFILE", "Type": "IncrementalArray", "Distribution": {"parameters": {"prefix": "p", "suffix": "s"}}}]
    assert d["optimization"]["parameters"]["optimizer"] == "DFTR"
    assert d["optimization"]["parameters"]["maxIter"] == 42

    restored = EnsembleConfig.from_dict(d)
    assert len(restored.parameters) == 1
    assert restored.parameters[0].name == "$PERMFILE"
    assert restored.optimization.optimizer == "DFTR"
    assert restored.optimization.max_iter == 42
    assert len(restored.controls) == 1
