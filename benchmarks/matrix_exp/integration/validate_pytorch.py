"""Numerical errors against stock PyTorch, for forward and autograd backward."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

from matrix_exp_bench.cases import DTYPE_MAP, make_input, reference
from matrix_exp_bench.reporting import write_error_summary

import torch


def errors(actual, expected):
    actual, expected = actual.detach().double(), expected.detach().double()
    delta = (actual - expected).abs().flatten(-2).amax(-1)
    magnitude = expected.abs().flatten(-2).amax(-1)
    actual_finite = actual.isfinite().flatten(-2).all(-1)
    reference_finite = expected.isfinite().flatten(-2).all(-1)
    finite = actual_finite & reference_finite
    normalized = torch.where(finite, delta / (1 + magnitude), float("inf"))
    finite_errors = normalized[finite]
    return dict(
        finite=bool(finite.all().item()),
        max_abs_error=delta.max().item(),
        global_normalized_error=(delta.max() / (1 + magnitude.max())).item(),
        worst_matrix_normalized_error=normalized.max().item(),
        median_finite_matrix_normalized_error=finite_errors.median().item()
        if finite_errors.numel()
        else float("nan"),
        p95_finite_matrix_normalized_error=torch.quantile(finite_errors, 0.95).item()
        if finite_errors.numel()
        else float("nan"),
        actual_nonfinite_matrices=int((~actual_finite).sum().item()),
        reference_nonfinite_matrices=int((~reference_finite).sum().item()),
        worst_matrix_index=normalized.argmax().item(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    implementation = parser.add_mutually_exclusive_group(required=True)
    implementation.add_argument("--library", type=Path)
    implementation.add_argument(
        "--native",
        action="store_true",
        help="Test the installed public API against its rank-4 fallback",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument(
        "--dtype", nargs="+", choices=DTYPE_MAP, default=list(DTYPE_MAP)
    )
    parser.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=[1, 32, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 120064],
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "numerical_summary.json"
    summary_path.unlink(missing_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    if args.native:
        forward = torch.linalg.matrix_exp

        def baseline_forward(a):
            return (
                torch.linalg.matrix_exp(a.float().unsqueeze(0)).squeeze(0).to(a.dtype)
            )
    else:
        from compare import MatrixExp

        torch.ops.load_library(str(args.library.resolve()))
        forward = MatrixExp.apply

        def baseline_forward(a):
            return torch.linalg.matrix_exp(a.float()).to(a.dtype)

    metadata = dict(
        host=platform.node(),
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        git=torch.version.git_version,
        torch_file=torch.__file__,
        library=str(args.library.resolve()) if args.library else None,
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest()
        if args.library
        else None,
        args={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        baseline=(
            "Same-build rank-4 fallback: torch.linalg.matrix_exp(a.float().unsqueeze(0)).squeeze(0).to(a.dtype)"
            if args.native
            else "torch.linalg.matrix_exp(a.float()).to(a.dtype), including autograd through casts"
        ),
        implementation="Installed public API and generated autograd"
        if args.native
        else "ATen extension probe",
        metric="max_abs_error_per_matrix / (1 + max_abs_reference_per_matrix)",
        fp64="All matrices in family sweep; first 32 matrices in timing-grid cases",
        grad="torch.autograd.grad with identical random upstream gradients",
        seeds={"input": 841, "upstream_gradient": 846},
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    cases = [("timing_grid", b, "large", 3.0) for b in args.batch]
    families = [
        (kind, 3.0)
        for kind in ("random", "skew", "neg_psd", "diagonal", "zero", "identity")
    ] + [("large", norm) for norm in (3.01006, 3.01007, 6.02014, 12.04028)]
    cases += [("families", b, kind, norm) for b in (35, 129) for kind, norm in families]
    rows = []
    fields = [
        "suite",
        "n",
        "dtype",
        "batch",
        "family",
        "norm",
        "operation",
        "comparison",
        "reference_policy",
        "compared_matrices",
        "finite",
        "max_abs_error",
        "global_normalized_error",
        "worst_matrix_normalized_error",
        "median_finite_matrix_normalized_error",
        "p95_finite_matrix_normalized_error",
        "actual_nonfinite_matrices",
        "reference_nonfinite_matrices",
        "worst_matrix_index",
    ]
    with (args.output / "numerical_errors.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for n in args.n:
            for dtype_name in args.dtype:
                dtype = DTYPE_MAP[dtype_name]
                for suite, batch, family, norm in cases:
                    a = make_input(batch, n, dtype, family, norm, seed=841)
                    ours = a.detach().requires_grad_()
                    stock = a.detach().requires_grad_()
                    generator = torch.Generator(device="cuda").manual_seed(846)
                    g = (
                        torch.randn(
                            a.shape, device="cuda", dtype=dtype, generator=generator
                        )
                        * 0.2
                    )
                    y = forward(ours)
                    expected = baseline_forward(stock)
                    (grad,) = torch.autograd.grad(y, ours, g)
                    (expected_grad,) = torch.autograd.grad(expected, stock, g)
                    sample = batch if suite == "families" else min(32, batch)
                    forward64 = reference(a[:sample])
                    backward64 = reference(a[:sample].mT, g[:sample])
                    for operation, actual, baseline, ref64 in (
                        ("forward", y, expected, forward64),
                        ("backward", grad, expected_grad, backward64),
                    ):
                        comparisons = [
                            (
                                "cuda_vs_pytorch",
                                actual,
                                baseline,
                                "native_fp32"
                                if dtype == torch.float32
                                else "fp32_with_input_output_casts",
                            ),
                            ("cuda_vs_fp64", actual[:sample], ref64, "fp64"),
                            (
                                "pytorch_vs_fp64",
                                baseline[:sample],
                                ref64,
                                "fp64",
                            ),
                        ]
                        for comparison, value, target, policy in comparisons:
                            measured = errors(value, target)
                            row = dict(
                                suite=suite,
                                n=n,
                                dtype=dtype_name,
                                batch=batch,
                                family=family,
                                norm=norm,
                                operation=operation,
                                comparison=comparison,
                                reference_policy=policy,
                                compared_matrices=len(value),
                                **measured,
                            )
                            writer.writerow(row)
                            rows.append(row)
                            f.flush()
                            # Retain the worst matrix of every comparison, without
                            # deciding which amount of degradation is acceptable.
                            if len(value):
                                index = measured["worst_matrix_index"]
                                path = (
                                    args.output
                                    / f"worst_{suite}_{n}_{dtype_name}_{batch}_{family}_{norm}_{operation}_{comparison}.pt"
                                )
                                torch.save(
                                    dict(
                                        input=a[index].detach().cpu(),
                                        upstream=g[index].detach().cpu(),
                                        actual=value[index].detach().cpu(),
                                        expected=target[index].detach().cpu(),
                                    ),
                                    path,
                                )
                    del a, ours, stock, g, y, expected, grad, expected_grad
                    print(n, dtype_name, suite, batch, family, norm, flush=True)
    write_error_summary(summary_path, rows, "worst_matrix_normalized_error")
    print(
        f"{len(rows)} numerical comparisons; see CSV and saved worst cases", flush=True
    )


if __name__ == "__main__":
    main()
