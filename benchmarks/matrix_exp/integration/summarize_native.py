"""Compare installed torch_small_n and torch_linalg wall timings."""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
from timing_metadata import (
    check_environment,
    check_shared_settings,
    load_timing_metadata,
    write_manifest,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    (output / "wall_manifest.json").unlink(missing_ok=True)
    grouped = defaultdict(lambda: defaultdict(list))
    raw = []
    environments = {}
    trial_records = []
    for path in sorted((output / "timings").glob("*_trial*.csv")):
        metadata = load_timing_metadata(path)
        backend = metadata["backend"]
        if backend in environments:
            check_environment(environments[backend], metadata)
        environments[backend] = metadata
        if not metadata.get("started_at_utc") or not metadata.get("completed_at_utc"):
            raise RuntimeError(f"Timing trial lacks completed-run timestamps: {path}")
        trial_records.append(
            dict(csv=str(path.relative_to(output)), environment=metadata)
        )
        with path.open() as stream:
            trial_rows = list(csv.DictReader(stream))
        for row in trial_rows:
            if row["backend"] != backend or int(row["trial"]) != metadata["trial"]:
                raise RuntimeError(f"Timing row does not match its metadata: {path}")
            if int(row["samples"]) != metadata["samples"]:
                raise RuntimeError(
                    f"Timing sample count does not match its metadata: {path}"
                )
            key = (int(row["n"]), row["dtype"], int(row["batch"]), row["operation"])
            grouped[key][row["backend"]].append(row)
            raw.append(row)
    if set(environments) != {"torch_small_n", "torch_linalg"}:
        raise RuntimeError("Both torch_small_n and torch_linalg trials are required")
    check_shared_settings(environments["torch_small_n"], environments["torch_linalg"])
    grid = environments["torch_small_n"]["grid"]
    expected = set(
        itertools.product(grid["n"], grid["dtype"], grid["batch"], grid["operation"])
    )
    if set(grouped) != expected:
        raise RuntimeError("Timing rows do not cover the declared grid")
    rows = []
    for (n, dtype, batch, operation), backends in sorted(grouped.items()):
        if set(backends) != {"torch_small_n", "torch_linalg"}:
            raise RuntimeError(
                f"Expected torch_small_n and torch_linalg, got {set(backends)}"
            )
        fingerprints = {
            (r["input_sha256"], r["upstream_sha256"])
            for data in backends.values()
            for r in data
        }
        if len(fingerprints) != 1:
            raise RuntimeError(f"Input mismatch at {(n, dtype, batch, operation)}")
        record = dict(n=n, dtype=dtype, batch=batch, operation=operation)
        for backend in ("torch_small_n", "torch_linalg"):
            trials = backends[backend]
            if sorted(int(r["trial"]) for r in trials) != [0, 1, 2]:
                raise RuntimeError(
                    f"Missing or duplicated trials: {(n, dtype, batch, operation, backend)}"
                )
            values = [float(r["median_ms"]) for r in trials]
            record[f"{backend}_median_ms"] = float(np.median(values))
        record["speedup"] = (
            record["torch_linalg_median_ms"] / record["torch_small_n_median_ms"]
        )
        rows.append(record)
    for filename, data in (("wall_summary.csv", rows), ("wall_timings.csv", raw)):
        with (output / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    labels = {
        "torch_small_n": "torch_small_n (our CUDA kernel)",
        "torch_linalg": f"torch_linalg (PyTorch {environments['torch_linalg']['torch']})",
    }
    for dtype in grid["dtype"]:
        for op in grid["operation"]:
            fig, axes = plt.subplots(
                1, len(grid["n"]), figsize=(4 * len(grid["n"]), 4.2), squeeze=False
            )
            axes = axes[0]
            for ax, n in zip(axes, grid["n"]):
                data = sorted(
                    (
                        r
                        for r in rows
                        if r["n"] == n and r["dtype"] == dtype and r["operation"] == op
                    ),
                    key=lambda r: r["batch"],
                )
                for backend in ("torch_linalg", "torch_small_n"):
                    ax.loglog(
                        [r["batch"] for r in data],
                        [r[f"{backend}_median_ms"] for r in data],
                        marker="o",
                        markersize=3,
                        label=labels[backend],
                    )
                ax.set_title(f"N={n}", y=1.02)
                ax.set_xlabel("Batch size")
                ax.grid(True, which="both", alpha=0.2)
            axes[0].set_ylabel("Synchronized wall time (ms)")
            axes[0].legend(fontsize=7)
            scope = (
                "forward" if op == "fwd" else "backward only (forward graph excluded)"
            )
            policy = "" if dtype == "fp32" else "; torch_linalg uses FP32 with casts"
            fig.suptitle(f"{dtype.upper()} {scope}{policy}")
            fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.8, wspace=0.35)
            for suffix in ("png", "pdf"):
                fig.savefig(
                    output / f"wall_{op}_{dtype}.{suffix}", dpi=160, bbox_inches="tight"
                )
            plt.close(fig)
    report = [
        "torch_small_n versus torch_linalg",
        "================================",
        "",
        "Public API measurements",
        "-----------------------",
        "",
        f"torch_small_n calls torch.linalg.matrix_exp and its generated autograd implementation in {environments['torch_small_n']['torch']}.",
        f"torch_linalg calls the default matrix exponential in {environments['torch_linalg']['torch']}.",
        "For BF16/FP16, torch_linalg uses FP32 matrix exponential and casts back, including autograd through those casts.",
        "torch_small_n uses FP32 scalar FMA intermediates for all input dtypes, with tiny kernels for N<=8 and cooperative scalar kernels above.",
        "Only the final output is converted back to the input dtype. The BF16 MMA path has been removed.",
        "Scaling is adaptive. Timings use prescribed matrix 1-norm 3; other norms can change the squaring count.",
        "",
        "Timings include allocations, conversions, Python dispatch/autograd, and CUDA synchronization.",
        f"Backward timings exclude preparation of the forward graph. Three trials use {environments['torch_small_n']['samples']} samples each.",
        "The summary reports the median of trial medians; raw samples and p99/variance remain in wall_timings.csv.",
        "The speedup column is torch_linalg_median_ms / torch_small_n_median_ms; values above one favor our kernel.",
        "Input and upstream-gradient hashes match across every compared backend and trial.",
        "Both backends load the same saved tensors created by the first torch_linalg run. Hostnames are recorded per timing row.",
        "",
    ]
    for dtype in grid["dtype"]:
        for op in grid["operation"]:
            data = [r for r in rows if r["dtype"] == dtype and r["operation"] == op]
            report.append(
                f"{dtype} {op}: speedup range {min(r['speedup'] for r in data):.2f}"
                f" to {max(r['speedup'] for r in data):.2f}x."
            )
            report.extend(["", f".. image:: wall_{op}_{dtype}.png", ""])
    report.extend(
        [
            "Limits",
            "------",
            "",
            f"Measurements apply to {environments['torch_small_n']['gpu']} and these inputs. The two installations differ in PyTorch revision,",
            "toolkit, and build features. wall_manifest.json records dates, source identities, binaries, settings and artifact hashes.",
            "The complete run_timings.sh workflow also writes accuracy/report.rst, per-matrix errors, empirical CDFs, and matched-input speedup/error figures.",
            "Numerical collection is fresh on every invocation; accuracy/figure_manifest.json identifies a complete numerical report tied to this timing run.",
        ]
    )
    (output / "report.rst").write_text("\n".join(report) + "\n")
    artifacts = [
        output / name for name in ("wall_summary.csv", "wall_timings.csv", "report.rst")
    ]
    for trial in trial_records:
        csv_path = output / trial["csv"]
        artifacts.extend((csv_path, csv_path.with_suffix(".json")))
    for dtype in grid["dtype"]:
        for op in grid["operation"]:
            artifacts.extend(
                output / f"wall_{op}_{dtype}.{suffix}" for suffix in ("png", "pdf")
            )
    write_manifest(output, trial_records, artifacts)
    print(
        json.dumps(
            dict(
                timing_cases=len(rows),
                raw_rows=len(raw),
                timed_samples=sum(int(r["samples"]) for r in raw),
            )
        )
    )


if __name__ == "__main__":
    main()
