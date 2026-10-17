import argparse
import csv
from pathlib import Path

from matrix_exp_bench import expm, frechet, THETA18
from matrix_exp_bench.cases import DTYPE_MAP, make_input, metrics, reference
from matrix_exp_bench.reporting import write_error_summary

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", nargs="+", type=int, default=[4, 8, 16, 32])
    p.add_argument("--dtype", nargs="+", choices=DTYPE_MAP, default=list(DTYPE_MAP))
    p.add_argument("--backends", nargs="+", default=["scalar"])
    p.add_argument(
        "--operations", nargs="+", choices=["fwd", "bwd"], default=["fwd", "bwd"]
    )
    p.add_argument("--batch", type=int, default=33)
    p.add_argument("--output", default="results/correctness.csv")
    args = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(2)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = path.with_suffix(".summary.json")
    # An interrupted run must not leave a completed summary from an earlier run.
    summary_path.unlink(missing_ok=True)
    writer = None
    rows = []
    key_fields = (
        "backend",
        "scaling",
        "S",
        "N",
        "B",
        "dtype",
        "kind",
        "norm",
        "operation",
        "category",
    )
    matrix_fields = list(key_fields) + [
        "matrix_index",
        "max_abs_err",
        "ref_max_abs",
        "matrix_rel_err",
        "finite",
    ]
    with (
        path.open("w", newline="") as stream,
        path.with_name(path.stem + "_per_matrix.csv").open(
            "w", newline=""
        ) as matrix_stream,
    ):
        matrix_writer = csv.DictWriter(matrix_stream, fieldnames=matrix_fields)
        matrix_writer.writeheader()
        for n in args.n:
            for dtype_name in args.dtype:
                dtype = DTYPE_MAP[dtype_name]
                cases = [
                    (k, 5.0)
                    for k in (
                        "random",
                        "skew",
                        "neg_psd",
                        "diagonal",
                        "zero",
                        "identity",
                    )
                ]
                cases += [
                    ("large", v)
                    for v in (
                        1.0,
                        3.0,
                        5.0,
                        THETA18 * (1 - 1e-5),
                        THETA18 * (1 + 1e-5),
                        2 * THETA18,
                        4 * THETA18 * (1 + 1e-5),
                        24.0,
                    )
                ]
                for kind, norm in cases:
                    a = make_input(args.batch, n, dtype, kind, norm)
                    gen = torch.Generator(device="cuda").manual_seed(812)
                    g = (
                        torch.randn(a.shape, device="cuda", dtype=dtype, generator=gen)
                        * 0.2
                    )
                    refs = {"fwd": reference(a), "bwd": reference(a.mT, g)}
                    for scaling in ("fixed", "adaptive"):
                        for backend in args.backends:
                            if backend == "triton_original" and scaling != "fixed":
                                continue
                            for operation in args.operations:
                                x = a if operation == "fwd" else a.mT
                                col = x.double().abs().sum(-2)
                                if operation == "bwd":
                                    col = col + g.double().abs().sum(-2)
                                norms = col.amax(-1)
                                ref = refs[operation]
                                representable = (
                                    torch.isfinite(ref.to(dtype)).flatten(-2).all(-1)
                                )
                                bound = (
                                    (norms <= THETA18 * 4)
                                    if scaling == "fixed"
                                    else torch.ones_like(norms, dtype=torch.bool)
                                )
                                eligible = bound & representable
                                got = (
                                    expm(a, backend, scaling)
                                    if operation == "fwd"
                                    else frechet(a.mT, g, backend, scaling)
                                )
                                for category, selection in (
                                    ("in_range", eligible),
                                    ("stress", ~eligible),
                                ):
                                    if not selection.any().item():
                                        continue
                                    # Output rounding is part of the contract, as in the original benchmarks.
                                    selected_ref = ref[selection].to(dtype)
                                    m = metrics(got[selection], selected_ref)
                                    row = dict(
                                        backend=backend,
                                        scaling=scaling,
                                        S=2,
                                        N=n,
                                        B=args.batch,
                                        dtype=dtype_name,
                                        kind=kind,
                                        norm=norm,
                                        operation=operation,
                                        category=category,
                                        tested_matrices=int(selection.sum().item()),
                                        **m,
                                    )
                                    indices = selection.nonzero().flatten().tolist()
                                    row["worst_matrix_index"] = indices[
                                        row["worst_matrix_index"]
                                    ]
                                    errors = (
                                        (
                                            got[selection].double()
                                            - selected_ref.double()
                                        )
                                        .abs()
                                        .flatten(-2)
                                        .amax(-1)
                                    )
                                    magnitudes = (
                                        selected_ref.double().abs().flatten(-2).amax(-1)
                                    )
                                    finite = (
                                        torch.isfinite(got[selection])
                                        .flatten(-2)
                                        .all(-1)
                                    )
                                    finite &= (
                                        torch.isfinite(selected_ref).flatten(-2).all(-1)
                                    )
                                    rels = torch.where(
                                        finite, errors / (1 + magnitudes), float("inf")
                                    )
                                    for idx, err, mag, rel, fin in zip(
                                        indices,
                                        errors.tolist(),
                                        magnitudes.tolist(),
                                        rels.tolist(),
                                        finite.tolist(),
                                    ):
                                        matrix_writer.writerow(
                                            {
                                                **{k: row[k] for k in key_fields},
                                                "matrix_index": idx,
                                                "max_abs_err": err,
                                                "ref_max_abs": mag,
                                                "matrix_rel_err": rel,
                                                "finite": fin,
                                            }
                                        )
                                    matrix_stream.flush()
                                    if writer is None:
                                        writer = csv.DictWriter(
                                            stream, fieldnames=list(row)
                                        )
                                        writer.writeheader()
                                    writer.writerow(row)
                                    stream.flush()
                                    rows.append(row)
                print(f"completed N={n} {dtype_name}", flush=True)
    write_error_summary(summary_path, rows, "worst_matrix_rel_err")
    print(f"{len(rows)} numerical comparisons; report: {path}")


if __name__ == "__main__":
    main()
