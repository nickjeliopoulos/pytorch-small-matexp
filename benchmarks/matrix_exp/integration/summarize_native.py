"""Compare installed torch_small_n and torch_linalg wall timings."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
from timing_metadata import check_environment, load_timing_metadata


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    grouped = defaultdict(lambda: defaultdict(list))
    raw = []
    environments = {}
    for path in sorted((output / "timings").glob("*_trial*.csv")):
        metadata = load_timing_metadata(path)
        backend = metadata["backend"]
        if backend in environments:
            check_environment(environments[backend], metadata)
        environments[backend] = metadata
        for row in csv.DictReader(path.open()):
            if row["backend"] != backend or int(row["trial"]) != metadata["trial"]:
                raise RuntimeError(f"Timing row does not match its metadata: {path}")
            key = (int(row["n"]), row["dtype"], int(row["batch"]), row["operation"])
            grouped[key][row["backend"]].append(row)
            raw.append(row)
    if len(grouped) != 264:
        raise RuntimeError(f"Expected 264 timing cases, found {len(grouped)}")
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
        "torch_linalg": "torch_linalg (PyTorch 2.14.1)",
    }
    for dtype in ("fp32", "bf16", "fp16"):
        for op in ("fwd", "bwd"):
            fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
            for ax, n in zip(axes, (4, 8, 16, 32)):
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
        "torch_small_n calls torch.linalg.matrix_exp and its generated autograd implementation in our custom 2.16.0a0 build.",
        "torch_linalg calls the default matrix exponential in the preserved 2.14.1+cu130 wheel.",
        "For BF16/FP16, torch_linalg uses FP32 matrix exponential and casts back, including autograd through those casts.",
        "torch_small_n uses FP32 scalar FMA intermediates for all input dtypes, with tiny kernels for N<=8 and cooperative scalar kernels above.",
        "Only the final output is converted back to the input dtype. The BF16 MMA path has been removed.",
        "Scaling is adaptive. Timings use prescribed matrix 1-norm 3; other norms can change the squaring count.",
        "",
        "Timings include allocations, conversions, Python dispatch/autograd, and CUDA synchronization.",
        "Backward timings exclude preparation of the forward graph. Three alternating backend trials use 25 samples each.",
        "The summary reports the median of trial medians; raw samples and p99/variance remain in wall_timings.csv.",
        "The speedup column is torch_linalg_median_ms / torch_small_n_median_ms; values above one favor our kernel.",
        "Input and upstream-gradient hashes match across every compared backend and trial.",
        "Both backends load the same saved tensors created by the first torch_linalg run. Hostnames are recorded per timing row.",
        "",
    ]
    for dtype in ("fp32", "bf16", "fp16"):
        for op in ("fwd", "bwd"):
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
            "Measurements apply to the allocated A100 and these inputs. The two installations differ in PyTorch revision,",
            "toolkit, and build features, as recorded in timings/*.json. This timing report does not rerun numerical validation.",
        ]
    )
    (output / "report.rst").write_text("\n".join(report) + "\n")
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
