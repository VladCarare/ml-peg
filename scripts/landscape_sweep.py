"""
Run one landscape search under overridden settings, saving everything.

Every decision the search takes is recorded and written out: basin hopping
steps with their verdicts, NEB bands accepted and rejected with both clash
measures, connection attempt outcomes, and transition state failures. The
resulting network is written too.

Usage::

    landscape_sweep.py <molecule> <start> <outdir> <label> [key=value ...]

Overridable keys: neighbours, attempts, min_sep, min_frac, bh_sep, cap,
bh_steps, model, models_file, allow_uncommitted.

``models_file`` loads the model from a registry in ml-peg's YAML format kept
outside the repository, for checkpoints that are not to be published.

The summary records the code, environment and machine the search ran with. The
search's behaviour lives in ml-peg and topsearch, so the run refuses to start
unless both are installed from a git commit and are the copies actually
imported; ``allow_uncommitted=1`` overrides that, and the record still says so.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib
from importlib import metadata
import json
import pathlib
import platform
import socket
import sys
import warnings

warnings.filterwarnings("ignore")

from ase import Atoms  # noqa: E402
from ase.io import read, write  # noqa: E402
import numpy as np  # noqa: E402
from topsearch.global_optimisation import basin_hopping as bh_module  # noqa: E402
from topsearch.global_optimisation.basin_hopping import BasinHopping  # noqa: E402
from topsearch.sampling.exploration import NetworkSampling  # noqa: E402
from topsearch.transition_states import nudged_elastic_band as neb_module  # noqa: E402
from topsearch.transition_states.nudged_elastic_band import (
    NudgedElasticBand,  # noqa: E402
)
from topsearch.utils.recorder import SearchRecorder  # noqa: E402

from ml_peg.calcs.landscapes.landscape_search import (
    calc_landscape_search as calc,  # noqa: E402
)
from ml_peg.models.get_models import load_models  # noqa: E402

#: Distributions recorded with every search. Absent ones are skipped, so one
#: list serves every backend environment.
RECORDED = (
    "ml-peg",
    "topsearch",
    "torch",
    "ase",
    "numpy",
    "scipy",
    "e3nn",
    "mace-torch",
    "xyme-mace",
    "fairchem-core",
    "uma-polar",
)

#: The distributions whose code decides what the search does, by import name.
SEARCH_CODE = {"ml-peg": "ml_peg", "topsearch": "topsearch"}


def _sha256(path: pathlib.Path) -> str:
    """
    Digest a file.

    Parameters
    ----------
    path
        File to digest.

    Returns
    -------
    str
        Its SHA-256, in hex.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _distribution(name: str) -> dict:
    """
    Describe an installed distribution: its version, and where it came from.

    Parameters
    ----------
    name
        Distribution name.

    Returns
    -------
    dict
        ``version``, and ``source`` -- the installer's direct_url record, which
        names the commit of a git install -- when there is one. Empty if the
        distribution is not installed.
    """
    try:
        dist = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {}
    record = {"version": dist.version}
    direct_url = dist.read_text("direct_url.json")
    if direct_url:
        record["source"] = json.loads(direct_url)
    return record


def _provenance(models_file: str | None) -> dict:
    """
    Record the code, environment and machine a search runs with.

    Parameters
    ----------
    models_file
        Registry the model is loaded from, when it is not ml-peg's own.

    Returns
    -------
    dict
        Host, Python, device, this script's digest, the registry and its digest,
        and the version and source of each recorded distribution.
    """
    import torch

    registry = pathlib.Path(models_file) if models_file else None
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "script_sha256": _sha256(pathlib.Path(__file__)),
        "models_file": str(registry) if registry else None,
        "models_file_sha256": _sha256(registry) if registry else None,
        "distributions": {
            name: record for name in RECORDED if (record := _distribution(name))
        },
    }


def _require_committed(provenance: dict) -> None:
    """
    Refuse to search with code that no commit identifies.

    Each of ml-peg and topsearch must be installed from a git commit, and must
    be the copy actually imported: an editable install, a local path or a
    PYTHONPATH overlay would run code the record does not describe.

    Parameters
    ----------
    provenance
        As ``_provenance`` returns it.

    Raises
    ------
    SystemExit
        If either is not installed from a commit, or is imported from elsewhere.
    """
    for name, module_name in SEARCH_CODE.items():
        source = provenance["distributions"].get(name, {}).get("source", {})
        commit = source.get("vcs_info", {}).get("commit_id")
        module_file = importlib.import_module(module_name).__file__
        imported_from = pathlib.Path(module_file).resolve()
        installed_at = pathlib.Path(metadata.distribution(name).locate_file(""))
        if not commit or not imported_from.is_relative_to(installed_at.resolve()):
            raise SystemExit(
                f"{name} is not running from a git commit: installed from "
                f"{source or 'nowhere recorded'}, imported from {imported_from}. "
                "Install it from a pinned commit, or pass allow_uncommitted=1."
            )


molecule_key, start_index = sys.argv[1], int(sys.argv[2])
outdir = pathlib.Path(sys.argv[3])
label = sys.argv[4]
overrides = dict(pair.split("=", 1) for pair in sys.argv[5:])
outdir.mkdir(parents=True, exist_ok=True)

model_name = overrides.get("model", "mace-off-23-L")
models_file = overrides.get("models_file")

# Checked before anything slow, so a run that would be refused fails at once.
provenance = _provenance(models_file)
if overrides.get("allow_uncommitted") != "1":
    _require_committed(provenance)

# Search settings live as module constants, read at call time, so overriding
# them here is enough; nothing in the search caches them at import.
if "neighbours" in overrides:
    calc.NEIGHBOURS = int(overrides["neighbours"])
if "attempts" in overrides:
    calc.MAX_CONNECTION_ATTEMPTS = int(overrides["attempts"])
if "cap" in overrides:
    calc.MAX_MINIMISATION_STEPS = int(overrides["cap"])
if "bh_steps" in overrides:
    calc.BASIN_HOPPING_STEPS = int(overrides["bh_steps"])
if "min_sep" in overrides:
    neb_module.MIN_IMAGE_SEPARATION = float(overrides["min_sep"])
if "min_frac" in overrides:
    neb_module.MIN_NONBONDED_FRACTION = float(overrides["min_frac"])
if "bh_sep" in overrides:
    bh_module.MIN_ATOM_SEPARATION = float(overrides["bh_sep"])

recorder = SearchRecorder()

# _search builds the search components itself, so the recorder is injected by
# defaulting the keyword on their constructors.
for cls in (BasinHopping, NudgedElasticBand, NetworkSampling):
    original = cls.__init__

    def patched(self, *args, _original=original, **kwargs):
        """Construct as before, with the shared recorder unless one is given."""
        kwargs.setdefault("recorder", recorder)
        return _original(self, *args, **kwargs)

    cls.__init__ = patched

molecule = next(m for m in calc.MOLECULES if m.key == molecule_key)
stem = molecule.xyz.rsplit(".", 1)[0]
conformer = read(calc._starts_dir() / f"{stem}_starts.xyz", index=":")[start_index]
symbols = conformer.get_chemical_symbols()

calculator = load_models([model_name], filepath=models_file)[model_name].get_calculator(
    precision="high"
)
potential = calc._AseCalculatorPotential(symbols, calculator)

settings = {
    "label": label,
    "molecule": molecule_key,
    "start": start_index,
    "model": model_name,
    "neighbours": calc.NEIGHBOURS,
    "attempts": calc.MAX_CONNECTION_ATTEMPTS,
    "cap": calc.MAX_MINIMISATION_STEPS,
    "bh_steps": calc.BASIN_HOPPING_STEPS,
    "min_sep": neb_module.MIN_IMAGE_SEPARATION,
    "min_frac": neb_module.MIN_NONBONDED_FRACTION,
    "bh_sep": bh_module.MIN_ATOM_SEPARATION,
    "models_file": models_file,
    "provenance": provenance,
}
print(json.dumps(settings), flush=True)

network = calc._search(
    symbols,
    conformer.get_positions().flatten(),
    potential,
    calc.SEED_OFFSETS[start_index],
)

# ---- everything the search found -------------------------------------------
frames = calc._frames(network, symbols, molecule_key, calc.SEED_OFFSETS[start_index])
for atoms in frames:
    atoms.info["start"] = start_index
if frames:
    write(outdir / f"{label}_found.extxyz", frames, format="extxyz")

# ---- everything the search threw away --------------------------------------
by_stage = Counter()
for stage_event in recorder.counts():
    by_stage[stage_event] = recorder.counts()[stage_event]


def _info(entry: dict) -> dict:
    """
    Make a recorder entry safe to store as extxyz info.

    extxyz info values cannot contain spaces, and an empty one swallows the key
    after it, so strings are squashed and blanks dropped.

    Parameters
    ----------
    entry
        One event from the search recorder.

    Returns
    -------
    dict
        The entry's fields, cleaned, without its structures.
    """
    clean = {}
    for key, value in entry.items():
        if key == "structures" or value is None:
            continue
        if isinstance(value, str):
            value = value.replace(" ", "_").replace('"', "")[:80]
            if not value:
                continue
        elif isinstance(value, (list, tuple)):
            value = "_".join(str(v) for v in value) or "none"
        clean[key] = value
    return clean


written = {}
for stage, event in list(by_stage):
    entries = [
        e
        for e in recorder.events
        if e["stage"] == stage and e["event"] == event and "structures" in e
    ]
    if not entries:
        continue
    out = []
    for entry in entries:
        for which, position in enumerate(entry["structures"]):
            flat = np.asarray(position).reshape(-1)
            # A band is many images; split it so every geometry is a frame
            n_atoms = len(symbols)
            for image in flat.reshape(-1, n_atoms * 3):
                atoms = Atoms(symbols, positions=image.reshape(-1, 3))
                atoms.info = _info(entry)
                atoms.info["which"] = which
                out.append(atoms)
    path = outdir / f"{label}_{stage}_{event}.extxyz"
    write(path, out, format="extxyz")
    written[f"{stage}/{event}"] = len(out)

summary = {
    **settings,
    "n_minima": int(network.n_minima),
    "n_transition_states": len(list(network.G.edges)),
    "events": {f"{s}/{e}": n for (s, e), n in by_stage.items()},
    "frames_written": written,
}
(outdir / f"{label}_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary), flush=True)
print("SWEEP_DONE", flush=True)
