"""Plot measured per-matrix disagreement and matched-input wall-time speedups."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
from measure_accuracy import load_manifest


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


DTYPES = ["fp32", "bf16", "fp16"]
COLORS = {"fp32": "#1764ab", "bf16": "#d06a12", "fp16": "#16866a"}
MARKERS = {4: "o", 8: "s", 16: "^", 32: "D"}
NAMES = {
    "random": "Random",
    "skew": "Skew-symmetric",
    "neg_psd": "Negative semidefinite",
    "diagonal": "Diagonal",
    "rotation": "Embedded rotation",
}
LINEAR_LIMIT = 1e-10
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def save(fig, stem, root):
    for extension in ["png", "pdf", "svg"]:
        fig.savefig(root / f"{stem}.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_error_axis(ax, which="x", upper=0.1):
    ticks = [0, 1e-8, 1e-6, 1e-4, 1e-2, 1, 100]
    ticks = [t for t in ticks if t <= upper]
    labels = ["0" if t == 0 else rf"$10^{{{int(np.log10(t))}}}$" for t in ticks]
    if which == "x":
        ax.set_xscale("symlog", linthresh=LINEAR_LIMIT, linscale=0.5)
        ax.set_xlim(0, upper)
        ax.set_xticks(ticks, labels)
        ax.set_ylim(0, 1.025)
        ax.set_yticks(np.linspace(0, 1, 6))
    else:
        ax.set_yscale("symlog", linthresh=LINEAR_LIMIT, linscale=0.5)
        ax.set_ylim(0, upper)
        ax.set_yticks(ticks, labels)
    ax.grid(True, which="major", alpha=0.22, linestyle=":")


def cdf(ax, values, label, color, upper):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if not len(finite):
        ax.plot([0, upper], [0, 0], label=label, color=color)
        return
    if finite.max() > upper:
        raise RuntimeError(
            "CDF axis would hide finite errors; increase its upper limit"
        )
    unique, counts = np.unique(finite, return_counts=True)
    cumulative = np.cumsum(counts) / len(values)
    ax.step(
        np.r_[0, unique, upper],
        np.r_[0, cumulative, cumulative[-1]],
        where="post",
        color=color,
        label=label,
        linewidth=2,
    )


def load_measurements(root, manifest_hash):
    metadata = {}
    for backend in ["torch_linalg", "torch_small_n"]:
        record = json.loads((root / f"{backend}_metadata.json").read_text())
        if (
            record.get("status") != "complete"
            or record.get("timing_manifest_sha256") != manifest_hash
        ):
            raise RuntimeError(f"Incomplete or stale accuracy collection for {backend}")
        metadata[backend] = record
    first, second = metadata["torch_linalg"], metadata["torch_small_n"]
    if (
        first["specs"] != second["specs"]
        or first["script_sha256"] != second["script_sha256"]
    ):
        raise RuntimeError(
            "Accuracy collections use different specifications or scripts"
        )
    summaries = json.loads((root / "case_summary.json").read_text())
    by_case = {(row["case_id"], row["operation"]): row for row in summaries}
    expected = {(i, op) for i in range(len(first["specs"])) for op in ["fwd", "bwd"]}
    if set(by_case) != expected or len(summaries) != len(expected):
        raise RuntimeError("Incomplete or duplicate accuracy summaries")
    per_case = defaultdict(list)
    with (root / "per_matrix_errors.csv").open() as f:
        for row in csv.DictReader(f):
            key = (int(row["case_id"]), row["operation"])
            if key not in by_case or int(row["matrix_index"]) != len(per_case[key]):
                raise RuntimeError(
                    "Missing, duplicate, or unexpected per-matrix error row"
                )
            per_case[key].append(float(row["relative_fro_error"]))
    grouped = {}
    for (i, operation), summary in by_case.items():
        spec = first["specs"][i]
        if any(summary[k] != v for k, v in spec.items()):
            raise RuntimeError("Accuracy summary does not match its declared case")
        values = np.asarray(per_case[i, operation])
        if (
            len(values) != spec["batch"]
            or np.any(np.isnan(values))
            or np.any(values < 0)
            or float(values.max()) != float(summary["max_relative_error"])
            or np.sum(values == 0) != summary["exact_matrices"]
            or np.sum(~np.isfinite(values)) != summary["unplottable_matrices"]
        ):
            raise RuntimeError("Incomplete or inconsistent per-matrix errors")
        key = (
            spec["suite"],
            operation,
            spec["n"],
            spec["dtype"],
            spec["family"],
            spec["target_norm"],
        )
        grouped[key] = values
    return metadata, summaries, grouped


def accuracy_points(output, manifest, summaries):
    trials = defaultdict(list)
    for trial in manifest["trials"]:
        with (output / trial["csv"]).open() as stream:
            for row in csv.DictReader(stream):
                trials[
                    int(row["n"]), row["dtype"], int(row["batch"]), row["operation"]
                ].append(row)
    with (output / "wall_summary.csv").open() as f:
        timings = {
            (int(r["n"]), r["dtype"], int(r["batch"]), r["operation"]): r
            for r in csv.DictReader(f)
        }
    points = []
    for case in summaries:
        if case["suite"] != "timing":
            continue
        key = case["n"], case["dtype"], case["batch"], case["operation"]
        matched = trials[key]
        expected = {(b, t) for b in ["torch_linalg", "torch_small_n"] for t in range(3)}
        if (
            len(matched) != 6
            or {(r["backend"], int(r["trial"])) for r in matched} != expected
        ):
            raise RuntimeError(
                "Accuracy point does not have six matching timing trials"
            )
        for row in matched:
            if any(row[k] != case[k] for k in ["input_sha256", "upstream_sha256"]):
                raise RuntimeError("Accuracy and timing inputs differ")
        timing = timings[key]
        for backend in ["torch_linalg", "torch_small_n"]:
            median = np.median(
                [float(r["median_ms"]) for r in matched if r["backend"] == backend]
            )
            if median != float(timing[f"{backend}_median_ms"]):
                raise RuntimeError("Timing summary differs from its trials")
        ratio = float(timing["torch_linalg_median_ms"]) / float(
            timing["torch_small_n_median_ms"]
        )
        if not math.isclose(ratio, float(timing["speedup"]), rel_tol=1e-12):
            raise RuntimeError("Incorrect timing speedup")
        point = {
            k: case[k]
            for k in [
                "n",
                "dtype",
                "batch",
                "operation",
                "family",
                "target_norm",
                "matrices",
                "exact_matrices",
                "unplottable_matrices",
                "max_relative_error",
                "finite_median",
                "finite_p95",
                "finite_p99",
                "input_sha256",
                "upstream_sha256",
            ]
        }
        point["max_relative_error"] = float(point["max_relative_error"])
        point.update(
            {
                k: float(timing[k])
                for k in [
                    "torch_small_n_median_ms",
                    "torch_linalg_median_ms",
                    "speedup",
                ]
            }
        )
        points.append(point)
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Timing report directory"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    root = output / "accuracy"
    (root / "figure_manifest.json").unlink(missing_ok=True)
    manifest = load_manifest(output)
    manifest_hash = hashlib.sha256(
        (output / "wall_manifest.json").read_bytes()
    ).hexdigest()
    grid = manifest["measurement_settings"]["grid"]
    dtypes = [d for d in DTYPES if d in grid["dtype"]]
    dimensions = ", ".join(map(str, grid["n"]))
    metadata_by_backend, summaries, grouped = load_measurements(root, manifest_hash)
    points = accuracy_points(output, manifest, summaries)
    timing_batch = points[0]["batch"]
    stock_version = metadata_by_backend["torch_linalg"]["torch"]
    gpu = manifest["measurement_settings"]["gpu"]
    family_batch = next(
        s["batch"]
        for s in metadata_by_backend["torch_linalg"]["specs"]
        if s["suite"] == "core"
    )

    def select(suite, operation, dtype=None, family=None, norm=None, n=None):
        parts = [
            v
            for k, v in grouped.items()
            if k[0] == suite
            and k[1] == operation
            and (dtype is None or k[3] == dtype)
            and (family is None or k[4] == family)
            and (norm is None or k[5] == norm)
            and (n is None or k[2] == n)
        ]
        return np.concatenate(parts) if parts else np.array([])

    with (root / "speedup_accuracy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    def scatter(ax, operation):
        subset = [r for r in points if r["operation"] == operation]
        finite_errors = [
            r["max_relative_error"]
            for r in subset
            if np.isfinite(r["max_relative_error"])
        ]
        max_y = max(finite_errors, default=LINEAR_LIMIT)
        upper = max(1e-2, 10 ** np.ceil(np.log10(max(max_y, LINEAR_LIMIT)) + 0.15))
        for p in subset:
            error = p["max_relative_error"]
            if not np.isfinite(error):
                error = upper / 2
                ax.annotate(
                    "nonfinite/undefined",
                    (p["speedup"], error),
                    xytext=(0, 9),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            ax.scatter(
                p["speedup"],
                error,
                facecolors=COLORS[p["dtype"]] if p["dtype"] == "fp16" else "none",
                marker=MARKERS[p["n"]],
                s={"fp32": 100, "bf16": 170, "fp16": 65}[p["dtype"]],
                edgecolors=COLORS[p["dtype"]],
                linewidths=1.6,
                zorder=3,
            )
        ax.axvline(1, color="#666666", linestyle="--", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlim(0.75, max(p["speedup"] for p in subset) * 1.35)
        format_error_axis(ax, which="y", upper=upper)
        min_y = min(finite_errors, default=0)
        if min_y > 0:
            ax.set_yscale("log")
            ax.set_ylim(min_y / 3, upper)
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10))
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.LogFormatterMathtext(base=10)
            )
        ax.set_xlabel("Wall-time speedup over torch_linalg (times)")
        ax.set_ylabel("Maximum per-matrix relative error")
        handles = [
            Line2D(
                [],
                [],
                linestyle="",
                marker=m,
                color="#444444",
                markersize=7,
                label=f"N = {n}",
            )
            for n, m in MARKERS.items()
            if n in grid["n"]
        ]
        ax.legend(handles=handles, loc="lower left", framealpha=0.95, ncol=2)
        ax.text(
            0.97,
            0.96,
            "Lower-right is better",
            ha="right",
            va="top",
            transform=ax.transAxes,
            color="#555555",
            fontsize=10,
        )

    for operation, title in [
        ("fwd", "Forward matrix exponential"),
        ("bwd", "Autograd backward"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(14.6, 6.3))
        for dtype in dtypes:
            values = select("core", operation, dtype)
            exact = np.mean(values == 0) * 100
            undefined = np.sum(~np.isfinite(values))
            label = f"{dtype.upper()}  ({exact:.1f}% exact; {undefined} undefined)"
            all_core = select("core", operation)
            finite_core = all_core[np.isfinite(all_core)]
            upper = max(0.1, float(finite_core.max()) * 2 if len(finite_core) else 0.1)
            cdf(axes[0], values, label, COLORS[dtype], upper)
        format_error_axis(axes[0], upper=upper)
        axes[0].set_xlabel("Per-matrix relative Frobenius error")
        axes[0].set_ylabel("Empirical CDF")
        axes[0].legend(loc="lower right", framealpha=0.95)
        axes[0].set_title(
            "(A) Error distribution across families and norms", loc="left"
        )
        scatter(axes[1], operation)
        axes[1].set_title("(B) Speedup versus relative error", loc="left")
        fig.suptitle(
            f"{title}: torch_small_n versus stock torch_linalg", fontsize=16, y=0.98
        )
        core_count = len(select("core", operation, dtypes[0]))
        fig.text(
            0.02,
            0.012,
            f"A: N = {dimensions}; four families; target 1-norm = 0.01, 0.1, 1, 3, 6, 10; {core_count:,} matrices/dtype.\n"
            f"B: B = {timing_batch:,}; random matrices with target 1-norm 3; same saved inputs as wall-time measurements.\n"
            "CDF axis is linear from 0 to 1e-10, logarithmic above; zeros are exact agreement. Relative error uses the cast-back stock output.",
            fontsize=10,
            color="#444444",
        )
        fig.tight_layout(rect=(0, 0.14, 1, 0.94), w_pad=3)
        save(fig, f"accuracy_overview_{operation}", root)

    for suite, families, norms in [
        ("core", ["random", "skew", "neg_psd", "diagonal"], [0.01, 0.1, 1, 3, 6, 10]),
        ("stress", ["skew", "neg_psd", "rotation"], [30, 100, 1000]),
    ]:
        norm_colors = plt.colormaps["viridis"](np.linspace(0.08, 0.83, len(norms)))
        for operation, title in [("fwd", "Forward"), ("bwd", "Autograd backward")]:
            values = select(suite, operation)
            finite_values = values[np.isfinite(values)]
            largest = finite_values.max() if len(finite_values) else LINEAR_LIMIT
            upper = max(0.01, 10 ** np.ceil(np.log10(max(largest, LINEAR_LIMIT)) + 0.1))
            fig, axes = plt.subplots(
                len(families),
                len(dtypes),
                figsize=(4.8 * len(dtypes), 3.05 * len(families)),
                sharex=True,
                sharey=True,
                squeeze=False,
            )
            for i, family in enumerate(families):
                for j, dtype in enumerate(dtypes):
                    ax = axes[i, j]
                    for norm, color in zip(norms, norm_colors):
                        cdf(
                            ax,
                            select(suite, operation, dtype, family, norm),
                            f"{norm:g}",
                            color,
                            upper,
                        )
                    format_error_axis(ax, upper=upper)
                    ax.set_title(f"{NAMES[family]} | {dtype.upper()}", fontsize=12)
                    selected = select(suite, operation, dtype, family)
                    bad = np.sum(~np.isfinite(selected))
                    ax.text(
                        0.97,
                        0.08,
                        f"{len(selected):,} matrices; {np.mean(selected == 0):.1%} exact\nNonfinite/undefined: {bad}",
                        ha="right",
                        transform=ax.transAxes,
                        fontsize=9,
                        color="#555555",
                    )
                    if j == 0:
                        ax.set_ylabel("Empirical CDF")
                    if i == len(families) - 1:
                        ax.set_xlabel("Relative Frobenius error")
            handles = [
                Line2D([], [], color=c, linewidth=2, label=f"{v:g}")
                for v, c in zip(norms, norm_colors)
            ]
            fig.legend(
                handles=handles,
                title="Target input matrix 1-norm",
                loc="upper center",
                bbox_to_anchor=(0.5, 0.965),
                ncol=len(norms),
                frameon=False,
            )
            fig.suptitle(
                f"{title} error distributions | {'norm stress cases' if suite == 'stress' else 'family and norm breakdown'}",
                fontsize=16,
                y=0.995,
            )
            fig.text(
                0.03,
                0.005,
                f"N = {dimensions}; {family_batch} matrices per (N, dtype, family, norm). Every sample remains in the CDF denominator.\n"
                "Axis is linear from 0 to 1e-10, logarithmic above; 0 is exact agreement. Nonfinite/undefined errors remain outside the finite axis.\n"
                f"Reference: stock PyTorch {stock_version}, FP32 then cast back for FP16/BF16; backward includes autograd through casts.",
                fontsize=10,
                color="#444444",
            )
            fig.tight_layout(rect=(0, 0.065, 1, 0.93), h_pad=2.3, w_pad=1.2)
            save(fig, f"cdf_{suite}_{operation}", root)

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.1))
    for ax, operation, title in zip(axes, ["fwd", "bwd"], ["Forward", "Backward only"]):
        scatter(ax, operation)
        ax.set_title(title, loc="left")
    handles = [
        Line2D(
            [],
            [],
            color=COLORS[d],
            marker="o",
            markerfacecolor=COLORS[d] if d == "fp16" else "none",
            linestyle="",
            label=d.upper(),
        )
        for d in dtypes
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
    )
    fig.text(
        0.03,
        0.012,
        f"B = {timing_batch:,}; target matrix 1-norm 3; same input tensors for accuracy and timing. {gpu}.\n"
        "Speedup = torch_linalg / torch_small_n synchronized wall time, including wrapper costs; median of three trial medians.\n"
        f"Backward excludes construction of the forward graph. Error is maximum relative Frobenius error among all {timing_batch:,} matrices.",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.94), w_pad=3)
    save(fig, "speedup_accuracy", root)

    description = []
    for suite in ["core", "stress", "analytic", "timing"]:
        for operation in ["fwd", "bwd"]:
            for dtype in dtypes:
                values = select(suite, operation, dtype)
                finite = values[np.isfinite(values)]
                description.append(
                    dict(
                        suite=suite,
                        operation=operation,
                        dtype=dtype,
                        matrices=len(values),
                        exact_fraction=float(np.mean(values == 0)),
                        unplottable=int(np.sum(~np.isfinite(values))),
                        finite_median=float(np.median(finite)) if len(finite) else None,
                        finite_p95=float(np.quantile(finite, 0.95))
                        if len(finite)
                        else None,
                        finite_p99=float(np.quantile(finite, 0.99))
                        if len(finite)
                        else None,
                        max_relative_error=float(values.max()),
                    )
                )
    description = [
        {
            k: str(v) if isinstance(v, float) and not math.isfinite(v) else v
            for k, v in row.items()
        }
        for row in description
    ]
    (root / "distribution_summary.json").write_text(
        json.dumps(description, indent=2, allow_nan=False)
    )
    report = [
        "Matrix exponential error distributions and speedup",
        "=================================================",
        "",
        f"Reference: stock PyTorch {stock_version}; custom build: {metadata_by_backend['torch_small_n']['torch']}.",
        f"GPU: {gpu}. Dimensions: {dimensions}. Dtypes: {', '.join(dtypes)}.",
        "",
        "Each CDF sample is one matrix. Relative error is ||ours - stock||_F / ||stock||_F, evaluated in FP64.",
        "The core sweep uses random, skew-symmetric, negative-semidefinite and diagonal matrices at target 1-norms 0.01,0.1,1,3,6,10.",
        "The separate stress sweep uses skew-symmetric, negative-semidefinite and embedded rotation cases at norms 30,100,1000.",
        "Zero and identity cases remain in the CSV and are excluded from the core CDF.",
        "For BF16/FP16, stock computes matrix_exp in FP32 then casts back; its backward includes autograd through those casts.",
        "Large exact-match fractions in low precision reflect output quantization; this comparison is agreement with stock, not an FP64 accuracy reference.",
        "Zero error stays at zero. CDF axes are linear up to 1e-10 and logarithmic above it. Nonfinite/undefined samples remain in each CDF denominator.",
        "Numerical errors have no acceptance threshold. See per_matrix_errors.csv and case_summary.json for every result.",
        "",
        f"Speedup points use B={timing_batch}, target 1-norm 3, and the exact saved timing inputs and upstream gradients.",
        "The ordinate is maximum per-matrix relative error. Speedup is torch_linalg / torch_small_n wall time, using the median of three trial medians.",
        "Wrapper costs and synchronization are included; backward excludes construction of its forward graph. These are eager public API measurements.",
        "Library identities, timing-artifact hashes, and all six timing-trial input hashes are checked before generating these figures.",
        "The two installations differ in PyTorch revision, toolkit, and build features; speedups describe these installations.",
        "",
    ]
    for path in sorted(root.glob("*.png")):
        report.extend([f".. image:: {path.name}", ""])
    (root / "report.rst").write_text("\n".join(report))
    artifacts = [
        root / "per_matrix_errors.csv",
        root / "speedup_accuracy.csv",
        root / "case_summary.json",
        root / "distribution_summary.json",
        root / "report.rst",
        output / "wall_manifest.json",
        output / "wall_summary.csv",
        root / "torch_linalg_metadata.json",
        root / "torch_small_n_metadata.json",
    ]
    artifacts += (
        list(root.glob("*.png")) + list(root.glob("*.pdf")) + list(root.glob("*.svg"))
    )
    figure_manifest = dict(
        status="complete",
        timing_manifest_sha256=manifest_hash,
        plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifacts={
            str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in artifacts
        },
    )
    (root / "figure_manifest.json").write_text(
        json.dumps(figure_manifest, indent=2, allow_nan=False)
    )
    print(json.dumps(description, indent=2))


if __name__ == "__main__":
    main()
