"""Synchronized forward/backward wall time versus default PyTorch."""

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from compare import MatrixExp
from matrix_exp_bench.cases import DTYPE_MAP, make_input

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
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
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--operation", nargs="+", choices=("fwd", "bwd"), default=["fwd", "bwd"]
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.ops.load_library(str(args.library.resolve()))
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        host=platform.node(),
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        library=str(args.library.resolve()),
        library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        args={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        input="Seed 841, prescribed matrix 1-norm 3 before dtype rounding",
        scope="perf_counter + CUDA synchronization; allocations and copies included",
        operations={
            "fwd": "Forward only, input does not require gradients",
            "bwd": "autograd.grad with retain_graph=True; forward graph prepared outside timing",
        },
        cute="Compiled LinearAlgebra.cpp entry body and CUDA DispatchStub in extension",
        pytorch_fp32="torch.linalg.matrix_exp(a), unmodified installed wheel",
        pytorch_lowp="torch.linalg.matrix_exp(a.float()).to(a.dtype); both conversions timed",
    )
    path = args.output / "wall_timings.csv"
    metadata_path = args.output / "runs.jsonl"
    if path.exists():
        if not metadata_path.exists():
            raise RuntimeError(
                "Existing timings have no provenance; use a new output directory"
            )
        previous = json.loads(metadata_path.read_text().splitlines()[-1])
        same_environment = all(
            previous[key] == metadata[key]
            for key in ("library_sha256", "torch", "gpu", "input", "pytorch_lowp")
        )
        if not same_environment or previous["args"]["samples"] != args.samples:
            raise RuntimeError(
                "Benchmark configuration changed; use a new output directory"
            )
    with metadata_path.open("a") as f:
        f.write(json.dumps(metadata) + "\n")
    existing = list(csv.DictReader(path.open())) if path.exists() else []
    completed = {
        (
            int(r["n"]),
            r["dtype"],
            int(r["batch"]),
            r.get("operation", "fwd"),
            int(r["trial"]),
            r["backend"],
        )
        for r in existing
    }
    fields = [
        "n",
        "dtype",
        "batch",
        "operation",
        "trial",
        "backend",
        "median_ms",
        "mean_ms",
        "p99_ms",
        "variance_ms2",
        "samples",
        "raw_ms",
    ]
    if existing and "operation" not in existing[0]:
        for row in existing:
            row["operation"] = "fwd"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(existing)
        temporary.replace(path)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not existing:
            writer.writeheader()
        for n in args.n:
            for dtype_name in args.dtype:
                for batch_index, batch in enumerate(args.batch):
                    stock_name = (
                        "pytorch" if dtype_name == "fp32" else "pytorch_promoted"
                    )
                    for operation in args.operation:
                        if all(
                            (n, dtype_name, batch, operation, trial, backend)
                            in completed
                            for trial in range(args.trials)
                            for backend in (stock_name, "cute")
                        ):
                            continue
                        a = make_input(
                            batch, n, DTYPE_MAP[dtype_name], "large", 3.0, seed=841
                        )
                        if operation == "fwd":
                            functions = {
                                "cute": lambda: torch.ops.matrix_exp_aten_probe.entry(
                                    a
                                ),
                                stock_name: (
                                    (lambda: torch.linalg.matrix_exp(a))
                                    if dtype_name == "fp32"
                                    else (
                                        lambda: torch.linalg.matrix_exp(a.float()).to(
                                            a.dtype
                                        )
                                    )
                                ),
                            }
                        else:
                            ours = a.detach().requires_grad_()
                            stock = a.detach().requires_grad_()
                            generator = torch.Generator(device="cuda").manual_seed(846)
                            g = (
                                torch.randn(
                                    a.shape,
                                    dtype=a.dtype,
                                    device="cuda",
                                    generator=generator,
                                )
                                * 0.2
                            )
                            y = MatrixExp.apply(ours)
                            stock_y = torch.linalg.matrix_exp(stock.float()).to(a.dtype)
                            functions = {
                                "cute": lambda: torch.autograd.grad(
                                    y, ours, g, retain_graph=True
                                ),
                                stock_name: lambda: torch.autograd.grad(
                                    stock_y, stock, g, retain_graph=True
                                ),
                            }
                        for fn in functions.values():
                            for _ in range(5):
                                fn()
                        torch.cuda.synchronize()
                        for trial in range(args.trials):
                            order = [stock_name, "cute"]
                            if (trial + batch_index) % 2:
                                order.reverse()
                            for backend in order:
                                if (
                                    n,
                                    dtype_name,
                                    batch,
                                    operation,
                                    trial,
                                    backend,
                                ) in completed:
                                    continue
                                fn = functions[backend]
                                samples = []
                                for _ in range(args.samples):
                                    begin = time.perf_counter()
                                    fn()
                                    torch.cuda.synchronize()
                                    samples.append((time.perf_counter() - begin) * 1000)
                                writer.writerow(
                                    dict(
                                        n=n,
                                        dtype=dtype_name,
                                        batch=batch,
                                        operation=operation,
                                        trial=trial,
                                        backend=backend,
                                        median_ms=float(np.median(samples)),
                                        mean_ms=float(np.mean(samples)),
                                        p99_ms=float(np.quantile(samples, 0.99)),
                                        variance_ms2=float(np.var(samples)),
                                        samples=len(samples),
                                        raw_ms=json.dumps(samples),
                                    )
                                )
                                f.flush()
                        if operation == "bwd":
                            del ours, stock, y, stock_y, g
                        print(n, dtype_name, batch, operation, flush=True)


if __name__ == "__main__":
    main()
