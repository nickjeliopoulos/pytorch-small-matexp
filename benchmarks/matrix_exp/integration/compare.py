"""Compare separately built ATen probes on identical inputs and timing scopes."""

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from matrix_exp_bench.cases import DTYPE_MAP, make_input, metrics, reference

import torch


OPS = torch.ops.matrix_exp_aten_probe


class MatrixExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return OPS.forward(a)

    @staticmethod
    def backward(ctx, g):
        (a,) = ctx.saved_tensors
        return OPS.differential(a, g, True)


def validate(args):
    saved = {}
    previous = (
        torch.load(args.reference_outputs, weights_only=True)
        if args.reference_outputs
        else None
    )
    rows = []
    failures = 0
    families = [
        (kind, 3.0)
        for kind in ("random", "skew", "neg_psd", "diagonal", "zero", "identity")
    ] + [("large", norm) for norm in (3.0, 3.01006, 3.01007, 6.02014, 12.04028)]
    for n in args.n:
        for dtype_name in args.dtype:
            dtype = DTYPE_MAP[dtype_name]
            for family, norm in families:
                a = make_input(35, n, dtype, family, norm, seed=841)
                generator = torch.Generator(device="cuda").manual_seed(846)
                g = (
                    torch.randn(
                        a.shape, dtype=dtype, device="cuda", generator=generator
                    )
                    * 0.2
                )
                for operation in ("forward", "backward"):
                    x = a if operation == "forward" else a.mT
                    direction = None if operation == "forward" else g
                    actual = (
                        OPS.forward(x)
                        if direction is None
                        else OPS.frechet(x, direction)
                    )
                    expected = reference(x, direction)
                    tol = 1e-3 if dtype == torch.float32 else 0.05
                    if operation == "backward":
                        tol *= 5
                    values = metrics(actual, expected)
                    passed = values["finite"] and values["worst_matrix_rel_err"] <= tol
                    key = f"{n}/{dtype_name}/{family}/{norm}/{operation}"
                    cpu = actual.cpu()
                    saved[key] = cpu
                    row = dict(
                        label=args.label,
                        n=n,
                        dtype=dtype_name,
                        family=family,
                        norm=norm,
                        operation=operation,
                        tolerance=tol,
                        passed=passed,
                        **values,
                    )
                    if previous is not None:
                        old = previous[key]
                        row["bitwise_equal"] = torch.equal(cpu, old)
                        row["baseline_max_abs_diff"] = (
                            (cpu.double() - old.double()).abs().max().item()
                        )
                    rows.append(row)
                    failures += not passed
    with (args.output / f"{args.label}_correctness.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    torch.save(saved, args.output / f"{args.label}_outputs.pt")
    print(
        f"{args.label}: {len(rows)} correctness cases, {failures} failures", flush=True
    )
    if failures:
        raise RuntimeError(f"{failures} correctness cases failed; see CSV")


def benchmark(args):
    rows = []
    for n in args.n:
        for dtype_name in args.dtype:
            for batch in args.batch:
                a = make_input(batch, n, DTYPE_MAP[dtype_name], "large", 3.0, seed=841)
                generator = torch.Generator(device="cuda").manual_seed(846)
                g = (
                    torch.randn(
                        a.shape, dtype=a.dtype, device="cuda", generator=generator
                    )
                    * 0.2
                )
                x = a.mT
                differentiable = a.detach().requires_grad_()

                def train():
                    return torch.autograd.grad(
                        MatrixExp.apply(differentiable), differentiable, g
                    )

                funcs = {
                    "forward": lambda: OPS.forward(a),
                    "backward": lambda: OPS.frechet(x, g),
                    "forward_backward": train,
                }
                for operation, fn in funcs.items():
                    for _ in range(5):
                        fn()
                    torch.cuda.synchronize()
                    count = 32 if batch <= 1024 else 8
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for _ in range(count):
                            fn()
                    for _ in range(3):
                        graph.replay()
                    torch.cuda.synchronize()
                    for scope in ("graph", "wall"):
                        samples = []
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        for _ in range(args.samples):
                            if scope == "graph":
                                start.record()
                                graph.replay()
                                end.record()
                                end.synchronize()
                                samples.append(start.elapsed_time(end) / count)
                            else:
                                begin = time.perf_counter()
                                fn()
                                torch.cuda.synchronize()
                                samples.append((time.perf_counter() - begin) * 1000)
                        rows.append(
                            dict(
                                label=args.label,
                                trial=args.trial,
                                n=n,
                                dtype=dtype_name,
                                batch=batch,
                                operation=operation,
                                scope=scope,
                                median_ms=float(np.median(samples)),
                                mean_ms=float(np.mean(samples)),
                                p99_ms=float(np.quantile(samples, 0.99)),
                                variance_ms2=float(np.var(samples)),
                                samples=len(samples),
                            )
                        )
                print(args.label, args.trial, n, dtype_name, batch, flush=True)
    path = args.output / f"{args.label}_timings_{args.trial}.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("validate", "benchmark"))
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-outputs", type=Path)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument(
        "--dtype", nargs="+", choices=DTYPE_MAP, default=list(DTYPE_MAP)
    )
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 32, 1024, 65536])
    parser.add_argument("--samples", type=int, default=25)
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
        scaling="adaptive",
        graph=(
            "GPU work of complete wrappers, including conversions/copies; "
            "allocations captured"
        ),
        wall=(
            "Complete wrapper including allocation, Python dispatch, synchronization "
            "and autograd for forward_backward"
        ),
    )
    with (args.output / "runs.jsonl").open("a") as f:
        f.write(json.dumps(metadata) + "\n")
    (validate if args.mode == "validate" else benchmark)(args)


if __name__ == "__main__":
    main()
