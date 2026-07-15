"""Typed configuration objects for petlab.

These replace the loose nested dicts that used to be passed around between
``create_ensemble.py`` / ``run_ensemble.py`` / ``extract_ensemble.py`` /
``optimize_ensemble.py`` / ``hm_ensemble.py``. Everything is still backed by
the same JSON config files used today (see ``data/**/*.json``) -- these
classes only give that JSON a typed, documented shape and a single place
(``from_dict``) that knows how to read it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any

from petlab.io import read_json, save_to_json


@dataclass
class ParameterSpec:
    """An uncertain parameter that is sampled once per ensemble member and
    substituted into the deck template.

    ``type`` is one of ``SingleValue``, ``RandomField``, ``IncrementalText``,
    ``IncrementalValue`` -- see ``petlab.deck`` for what each does.
    """

    name: str
    type: str
    distribution: dict[str, Any]

    @classmethod
    def from_dict(cls, d: dict) -> "ParameterSpec":
        return cls(name=d["Name"], type=d["Type"], distribution=d["Distribution"])

    def to_dict(self) -> dict:
        return {"Name": self.name, "Type": self.type, "Distribution": self.distribution}


@dataclass
class ControlSpec:
    """A decision variable substituted into the deck (well control)."""

    name: str
    default: float
    kind: str = "float"  # "float" | "int" -- named `kind` to avoid shadowing builtins
    lb: float = -math.inf
    ub: float = math.inf

    @classmethod
    def from_dict(cls, d: dict) -> "ControlSpec":
        return cls(
            name=d["Name"],
            default=d["Default"],
            kind=d.get("type", "float"),
            lb=d.get("lb", -math.inf),
            ub=d.get("ub", math.inf),
        )

    def to_dict(self) -> dict:
        return {"Name": self.name, "Default": self.default, "type": self.kind, "lb": self.lb, "ub": self.ub}


@dataclass
class VectorsConfig:
    """Which simulator output vectors to extract after each run."""

    summary: list[str] = field(default_factory=list)
    static3d: list[str] = field(default_factory=list)
    dynamic3d: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> "VectorsConfig":
        d = d or {}
        return cls(
            summary=list(d.get("summary", [])),
            static3d=list(d.get("static3d", [])),
            dynamic3d=list(d.get("dynamic3d", [])),
        )

    def to_dict(self) -> dict:
        return {"summary": self.summary, "static3d": self.static3d, "dynamic3d": self.dynamic3d}


@dataclass
class HistoryMatchConfig:
    """History-matching settings. Default method is ESMDA (with PLSR
    dimensionality reduction baked in when the parameter matrix is wide --
    see ``petlab.historymatch.esmda``)."""

    method: str = "ESMDA"
    updatepath: str = ""
    model3d: list[str] = field(default_factory=list)
    timestep: str = ""
    objectives: dict[str, str] = field(default_factory=dict)
    ncomponent: int = 10
    alphas: list[float] = field(default_factory=lambda: [9.333, 7.0, 4.0, 2.0])
    polynomial_order: int = 3

    @classmethod
    def from_dict(cls, d: dict | None) -> "HistoryMatchConfig":
        d = d or {}
        return cls(
            method=d.get("method", "ESMDA"),
            updatepath=d.get("updatepath", ""),
            model3d=list(d.get("model3d", [])),
            timestep=d.get("timestep", ""),
            objectives=dict(d.get("objectives", {})),
            ncomponent=d.get("ncomponent", 10),
            alphas=list(d.get("alphas", [9.333, 7.0, 4.0, 2.0])),
            polynomial_order=d.get("polynomial_order", 3),
        )

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "updatepath": self.updatepath,
            "model3d": self.model3d,
            "timestep": self.timestep,
            "objectives": self.objectives,
            "ncomponent": self.ncomponent,
            "alphas": self.alphas,
            "polynomial_order": self.polynomial_order,
        }


@dataclass
class OptimizationConfig:
    """Optimization settings. Default optimizer is DFTR (the derivative-free
    trust-region SQP-filter method, ``py_trsqp``)."""

    optimizer: str = "DFTR"
    cost_function: str = "NPV"
    max_iter: int = 100
    constraints: dict[str, dict] = field(default_factory=dict)
    constants: dict[str, float] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    control_snap_pairs: list[list[int]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> "OptimizationConfig":
        d = d or {}
        params = d.get("parameters", d)  # accept both {"parameters": {...}} and flat dict
        return cls(
            optimizer=params.get("optimizer", "DFTR"),
            cost_function=params.get("costFunction", "NPV"),
            max_iter=params.get("maxIter", 100),
            constraints=dict(params.get("constraints", {})),
            constants=dict(params.get("constants", {})),
            options=dict(params.get("options", {})),
            control_snap_pairs=[list(p) for p in params.get("control_snap_pairs", [])],
        )

    def to_dict(self) -> dict:
        return {
            "parameters": {
                "optimizer": self.optimizer,
                "costFunction": self.cost_function,
                "maxIter": self.max_iter,
                "constraints": self.constraints,
                "constants": self.constants,
                "options": self.options,
                "control_snap_pairs": self.control_snap_pairs,
            }
        }


@dataclass
class EnsembleConfig:
    """Everything needed to create, run, extract and optimize one ensemble.

    This is the typed equivalent of the JSON files under ``data/**/*.json``.
    """

    name: str
    ne: int
    root: str
    parameters: list[ParameterSpec]
    controls: list[ControlSpec]
    n_parallel: int = 1
    vectors: VectorsConfig = field(default_factory=VectorsConfig)
    historymatching: HistoryMatchConfig = field(default_factory=HistoryMatchConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "EnsembleConfig":
        return cls(
            name=d["Name"],
            ne=d["Ne"],
            root=d["root"],
            n_parallel=d.get("n_parallel", 1),
            parameters=[ParameterSpec.from_dict(p) for p in d.get("parameters", [])],
            controls=[ControlSpec.from_dict(c) for c in d.get("controls", [])],
            vectors=VectorsConfig.from_dict(d.get("vectors")),
            historymatching=HistoryMatchConfig.from_dict(d.get("historymatching")),
            optimization=OptimizationConfig.from_dict(d.get("optimization")),
        )

    @classmethod
    def from_json(cls, path: str) -> "EnsembleConfig":
        return cls.from_dict(read_json(path))

    def to_dict(self) -> dict:
        return {
            "Name": self.name,
            "Ne": self.ne,
            "root": self.root,
            "n_parallel": self.n_parallel,
            "parameters": [p.to_dict() for p in self.parameters],
            "controls": [c.to_dict() for c in self.controls],
            "vectors": self.vectors.to_dict(),
            "historymatching": self.historymatching.to_dict(),
            # note: `self.optimization.to_dict()` is itself
            # `{"parameters": {...}}` (the legacy on-disk shape), which must
            # stay nested under its own "optimization" key here -- spreading
            # it at this level would clobber the *ensemble's* "parameters"
            # (the ParameterSpec list) with the optimizer's settings dict,
            # since they happen to share that key name in the on-disk schema.
            "optimization": self.optimization.to_dict(),
        }

    def to_json(self, path: str) -> None:
        save_to_json(path, self.to_dict())


@dataclass
class CLRMStageConfig:
    """One receding-horizon checkpoint of a CLRM run."""

    checkpoint_years: float
    control_indices: list[int]


@dataclass
class CLRMConfig:
    """Configuration for a full closed-loop (optimize -> apply to truth ->
    history-match -> repeat) run.

    ``prior`` is the N-member ensemble config (e.g. Egg realizations 1-16).
    ``truth`` is a single-realization config standing in for the real
    reservoir (e.g. Egg realization 100) that the optimized controls are
    actually applied to.
    """

    name: str
    prior: EnsembleConfig
    truth: EnsembleConfig
    truth_case_number: int
    stages: list[CLRMStageConfig]
    updated_parameter: str
    output_dir: str = "simulations/clrm"

    @classmethod
    def from_dict(cls, d: dict) -> "CLRMConfig":
        prior = EnsembleConfig.from_dict(d["prior"])
        truth = EnsembleConfig.from_dict(d["truth"])
        stages = [
            CLRMStageConfig(checkpoint_years=s["checkpoint_years"], control_indices=list(s["control_indices"]))
            for s in d["stages"]
        ]
        return cls(
            name=d["Name"],
            prior=prior,
            truth=truth,
            truth_case_number=d["truth_case_number"],
            stages=stages,
            updated_parameter=d["updated_parameter"],
            output_dir=d.get("output_dir", "simulations/clrm"),
        )

    @classmethod
    def from_json(cls, path: str) -> "CLRMConfig":
        return cls.from_dict(read_json(path))
