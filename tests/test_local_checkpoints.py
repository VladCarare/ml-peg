"""Tests for loading a UMA model from a local checkpoint, not a published name."""

from __future__ import annotations

import dataclasses
import sys
import types
from typing import Any

import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import FairChemCalc


@dataclasses.dataclass
class _Settings:
    """Stand-in for fairchem's InferenceSettings, with only the fields read here."""

    merge_mole: bool
    compile: bool
    base_precision_dtype: str = "float32"


class _Recorder:
    """Records how fairchem was called, so the test can assert on the route taken."""

    def __init__(self) -> None:
        self.load_predict_unit_calls: list[dict[str, Any]] = []
        self.get_predict_unit_calls: list[dict[str, Any]] = []
        self.calculator_calls: list[dict[str, Any]] = []


@pytest.fixture
def fairchem(monkeypatch) -> _Recorder:
    """Install a fake `fairchem.core` so the loader can be exercised without it."""
    recorder = _Recorder()

    def load_predict_unit(path, **kwargs):
        recorder.load_predict_unit_calls.append({"path": path, **kwargs})
        return f"predictor-from-{path}"

    def get_predict_unit(model_name, **kwargs):
        recorder.get_predict_unit_calls.append({"model_name": model_name, **kwargs})
        return f"predictor-for-{model_name}"

    def calculator(predictor, **kwargs):
        recorder.calculator_calls.append({"predictor": predictor, **kwargs})
        return f"calculator({predictor})"

    core = types.ModuleType("fairchem.core")
    core.FAIRChemCalculator = calculator
    core.pretrained_mlip = types.SimpleNamespace(
        get_predict_unit=get_predict_unit,
        _MODEL_CKPTS=types.SimpleNamespace(checkpoints={"uma-s-1p1"}),
    )

    unit = types.ModuleType("fairchem.core.units.mlip_unit")
    unit.load_predict_unit = load_predict_unit

    inference = types.ModuleType("fairchem.core.units.mlip_unit.api.inference")
    inference.inference_settings_default = lambda: _Settings(
        merge_mole=True, compile=True
    )
    inference.inference_settings_traineval = lambda: _Settings(
        merge_mole=False, compile=False
    )

    api = types.ModuleType("fairchem.core.units.mlip_unit.api")
    api.inference = inference
    units = types.ModuleType("fairchem.core.units")
    fairchem_pkg = types.ModuleType("fairchem")
    fairchem_pkg.core = core

    for name, module in (
        ("fairchem", fairchem_pkg),
        ("fairchem.core", core),
        ("fairchem.core.units", units),
        ("fairchem.core.units.mlip_unit", unit),
        ("fairchem.core.units.mlip_unit.api", api),
        ("fairchem.core.units.mlip_unit.api.inference", inference),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    return recorder


def _calc(**kwargs) -> FairChemCalc:
    """Build a FairChemCalc with the fields every case needs."""
    return FairChemCalc(
        task_name="omol", trained_on_dispersion=True, device="cpu", **kwargs
    )


def test_checkpoint_is_loaded_from_the_path(fairchem) -> None:
    """A local checkpoint goes through load_predict_unit, not the published registry."""
    _calc(checkpoint="/scratch/ckpt.pt").get_calculator(precision="high")

    assert [call["path"] for call in fairchem.load_predict_unit_calls] == [
        "/scratch/ckpt.pt"
    ]
    assert fairchem.get_predict_unit_calls == []


def test_a_published_name_still_uses_the_registry(fairchem) -> None:
    """The existing route is untouched by the addition of the checkpoint one."""
    _calc(model_name="uma-s-1p1").get_calculator(precision="low")

    assert [call["model_name"] for call in fairchem.get_predict_unit_calls] == [
        "uma-s-1p1"
    ]
    assert fairchem.load_predict_unit_calls == []


def test_checkpoint_disables_compile_and_merge(fairchem) -> None:
    """Both silently corrupt a locally trained checkpoint, so neither may be left on.

    `compile` mangles the polar head's two-stage gradient and inflates forces by
    about 1.8x; `merge_mole` makes a checkpoint trained with `regress_stress:
    False` raise "One of the differentiated Tensors does not require grad" on
    every structure through ASE.
    """
    _calc(checkpoint="/scratch/ckpt.pt").get_calculator(precision="high")

    settings = fairchem.load_predict_unit_calls[0]["inference_settings"]
    assert settings.compile is False
    assert settings.merge_mole is False


def test_checkpoint_honours_the_requested_precision(fairchem) -> None:
    """Precision must reach fairchem, which defaults to float32.

    float64 is not optional for the polar branch.
    """
    _calc(checkpoint="/scratch/ckpt.pt").get_calculator(precision="high")
    assert (
        fairchem.load_predict_unit_calls[0]["inference_settings"].base_precision_dtype
        == "float64"
    )

    _calc(checkpoint="/scratch/ckpt.pt").get_calculator(precision="low")
    assert (
        fairchem.load_predict_unit_calls[1]["inference_settings"].base_precision_dtype
        == "float32"
    )


def test_checkpoint_passes_overrides_through(fairchem) -> None:
    """`regress_stress` is baked in at training time and has to be overridden."""
    overrides = {"backbone": {"regress_stress": True}}
    _calc(checkpoint="/scratch/ckpt.pt", overrides=overrides).get_calculator(
        precision="high"
    )

    assert fairchem.load_predict_unit_calls[0]["overrides"] == overrides


def test_task_name_reaches_the_calculator(fairchem) -> None:
    """One checkpoint can serve two levels of theory, so the task is identity."""
    FairChemCalc(
        checkpoint="/scratch/ckpt.pt",
        task_name="omc",
        trained_on_dispersion=True,
        device="cpu",
    ).get_calculator(precision="high")

    assert fairchem.calculator_calls[0]["task_name"] == "omc"


def test_availability_is_the_file_existing(tmp_path, fairchem) -> None:
    """A checkpoint cannot be looked up in the published registry, so check the disk."""
    present = tmp_path / "ckpt.pt"
    present.write_bytes(b"")

    assert _calc(checkpoint=str(present)).available is True
    assert _calc(checkpoint=str(tmp_path / "absent.pt")).available is False


def test_registry_entry_routes_a_checkpoint(fairchem, tmp_path) -> None:
    """A `checkpoint` in the YAML reaches the calculator without a model name."""
    config = tmp_path / "models.yml"
    config.write_text(
        "local-uma:\n"
        "  module: fairchem.core\n"
        "  class_name: FAIRChemCalculator\n"
        "  trained_on_dispersion: true\n"
        "  kwargs:\n"
        "    checkpoint: /scratch/ckpt.pt\n"
        "    task_name: omol\n",
        encoding="utf8",
    )

    model = load_models(["local-uma"], filepath=config)["local-uma"]

    assert model.checkpoint == "/scratch/ckpt.pt"
    assert model.model_name is None


def test_a_published_entry_still_requires_a_model_name(tmp_path) -> None:
    """Dropping the requirement entirely would let a nameless entry load silently."""
    config = tmp_path / "models.yml"
    config.write_text(
        "nameless:\n"
        "  module: fairchem.core\n"
        "  class_name: FAIRChemCalculator\n"
        "  trained_on_dispersion: true\n"
        "  kwargs:\n"
        "    task_name: omol\n",
        encoding="utf8",
    )

    with pytest.raises(ValueError, match="model_name"):
        load_models(["nameless"], filepath=config)
