"""
Score one model's landscape searches on every subset of its starting conformers.

A landscape search is not reproducible, so the result from one start is a
single draw rather than a measurement. Scoring every subset of the starts gives
each start count a spread, which is what says whether five starts are enough.

The committed analysis runs unchanged; only where it reads and writes moves.
Each subset is linked into a temporary tree that becomes the analysis's
``CALC_PATH``, and ``OUT_PATH`` points at a temporary directory, so the
distorted-minima writer cannot touch the app's data.

Usage::

    landscape_seed_curve.py <model> <calc_path> <out.jsonl> [workers=N]

``calc_path`` holds ``<model>/<molecule>_start<N>_found.extxyz``, the layout the
analysis reads. Writes one JSON line per subset, then prints each metric's
spread per number of starts. Refuses to run unless the analysis is imported
from a git checkout without uncommitted changes, and records its commit.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

from ml_peg.analysis.landscapes.landscape_search import (  # noqa: E402
    analyse_landscape_search as analysis,
)

#: Pooled metrics, as (name, numerator, denominator) over the per-molecule
#: results, matching the analysis's own table.
METRICS = (
    ("minima_recovered", "minima_matched", "n_reference_minima"),
    ("saddles_recovered", "saddles_matched", "n_reference_saddles"),
    ("connectivity", "connectivity_preserved", "n_reference_saddles"),
    ("off_reference", "off_reference_minima", "n_found_minima"),
    ("distorted", "distorted_minima", "n_found_minima"),
    ("near_missed", "minima_near_missed", "n_reference_minima"),
)
START = re.compile(r"_start(\d+)_found\.extxyz$")


def _commit() -> str:
    """Return the commit the analysis is imported from, refusing local changes."""
    where = str(Path(analysis.__file__).parent)
    try:
        head, dirty = (
            subprocess.run(
                ["git", "-C", where, *args], capture_output=True, text=True, check=True
            ).stdout.strip()
            for args in (
                ("rev-parse", "HEAD"),
                ("status", "--porcelain", "--untracked-files=no"),
            )
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        sys.exit(f"{where} is not in a git checkout; run from a clean clone.")
    if dirty:
        sys.exit(f"uncommitted changes in the checkout of {where}:\n{dirty}")
    return head


def _start(path: Path) -> int:
    """Return the starting-conformer index in a search's output file name."""
    return int(START.search(path.name).group(1))


def _score(model: str, calc_path: Path, starts: tuple[int, ...]) -> dict:
    """Run the analysis on the searches from one subset of starts."""
    with tempfile.TemporaryDirectory() as scratch:
        linked = Path(scratch) / "calc" / model
        linked.mkdir(parents=True)
        for path in (calc_path / model).glob("*_found.extxyz"):
            if _start(path) in starts:
                (linked / path.name).symlink_to(path.resolve())
        analysis.CALC_PATH = linked.parent
        analysis.OUT_PATH = Path(scratch) / "out"
        results = analysis._compare(model)
    record = {"model": model, "starts": list(starts), "n_starts": len(starts)}
    for name, numerator, denominator in METRICS:
        record[name] = analysis._pooled(results, numerator, denominator)
    record["n_found_minima"] = sum(r["n_found_minima"] for r in results.values())
    record["n_found_saddles"] = sum(r["n_found_saddles"] for r in results.values())
    return record


def _summarise(records: list[dict]) -> None:
    """Print each metric's mean and range over subsets, per number of starts."""
    names = [name for name, _, _ in METRICS]
    print("starts subsets  " + "  ".join(f"{name:>23}" for name in names))
    for count in sorted({r["n_starts"] for r in records}):
        group = [r for r in records if r["n_starts"] == count]
        cells = []
        for name in names:
            values = [r[name] for r in group]
            mean = sum(values) / len(values)
            cells.append(f"{mean:.3f} [{min(values):.3f}-{max(values):.3f}]")
        print(f"{count:>6} {len(group):>7}  " + "  ".join(f"{c:>23}" for c in cells))


def main() -> None:
    """Score every subset of one model's starts and write the results."""
    model, calc_path, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    options = dict(arg.split("=", 1) for arg in sys.argv[4:])
    commit = _commit()
    starts = sorted({_start(p) for p in (calc_path / model).glob("*_found.extxyz")})
    subsets = [s for k in range(1, len(starts) + 1) for s in combinations(starts, k)]
    workers = int(options.get("workers", min(len(subsets), os.cpu_count() or 1)))
    records = []
    with ProcessPoolExecutor(workers) as pool, out.open("w") as handle:
        for record in pool.map(
            _score, [model] * len(subsets), [calc_path] * len(subsets), subsets
        ):
            record["analysis_commit"] = commit
            record["analysis"] = analysis.__file__
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            records.append(record)
    _summarise(records)


if __name__ == "__main__":
    main()
