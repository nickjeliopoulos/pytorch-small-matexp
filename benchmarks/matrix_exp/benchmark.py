"""Resumable raw-kernel and complete-wrapper measurements."""

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import triton
import triton.testing
from matrix_exp_bench import expm, frechet, raw
from matrix_exp_bench.cases import DTYPE_MAP, make_input

import torch


def functions(a, g, backend, scaling):
    x, xt, gf = (
        a.float().contiguous(),
        a.mT.float().contiguous(),
        g.float().contiguous(),
    )
    out, dout = torch.empty_like(a), torch.empty_like(a)
    ag = a.detach().requires_grad_(True)
    stock_y = expm(ag, backend, scaling) if backend == "torch" else None

    def train():
        y = expm(ag, backend, scaling)
        return torch.autograd.grad(y, ag, g)

    def backward():
        return torch.autograd.grad(stock_y, ag, g, retain_graph=True)

    def both_raw():
        raw(x, None, out, backend, scaling)
        raw(xt, gf, dout, backend, scaling)

    return {
        ("raw", "forward"): lambda: raw(x, None, out, backend, scaling),
        ("raw", "backward"): lambda: raw(xt, gf, dout, backend, scaling),
        ("raw", "forward_backward"): both_raw,
        ("wrapper", "forward"): lambda: expm(a, backend, scaling),
        ("wrapper", "backward"): backward
        if backend == "torch"
        else lambda: frechet(a.mT, g, backend, scaling),
        ("wrapper", "forward_backward"): train,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 1024, 65536])
    parser.add_argument(
        "--dtype", nargs="+", choices=DTYPE_MAP, default=["bf16", "fp16"]
    )
    parser.add_argument(
        "--backends", nargs="+", default=["scalar", "mma", "triton_original", "torch"]
    )
    parser.add_argument("--scaling", choices=["fixed", "adaptive"], default="fixed")
    parser.add_argument("--rep-ms", type=float, default=500.0)
    parser.add_argument("--output", type=Path, default=Path("results/timings.csv"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    keys = ["N", "B", "dtype", "backend", "scaling", "scope", "operation", "trial"]
    fields = keys + [
        "median_ms",
        "mean_ms",
        "p99_ms",
        "variance_ms2",
        "samples",
        "first_call_ms",
        "error",
    ]
    done = (
        {
            tuple(r[k] for k in keys)
            for r in csv.DictReader(args.output.open())
            if not r["error"]
        }
        if args.output.exists()
        else set()
    )
    metadata = dict(
        args={**vars(args), "output": str(args.output)},
        host=platform.node(),
        torch=torch.__version__,
        triton=triton.__version__,
        gpu=torch.cuda.get_device_name(),
    )
    with args.output.with_suffix(".runs.jsonl").open("a") as f:
        f.write(json.dumps(metadata) + "\n")
    failures = []
    with args.output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if args.output.stat().st_size == 0:
            writer.writeheader()
        for n in args.n:
            for dtype in args.dtype:
                for b in args.batch:
                    a = make_input(b, n, DTYPE_MAP[dtype], "large", 3.0)
                    gen = torch.Generator(device="cuda").manual_seed(812)
                    g = (
                        torch.randn(
                            a.shape, device=a.device, dtype=a.dtype, generator=gen
                        )
                        * 0.2
                    )
                    for trial in range(3):
                        for backend in (
                            args.backends if trial % 2 == 0 else args.backends[::-1]
                        ):
                            if (
                                args.scaling == "adaptive"
                                and backend == "triton_original"
                            ):
                                continue
                            funcs = functions(a, g, backend, args.scaling)
                            for (scope, op), fn in funcs.items():
                                if scope == "raw" and backend == "torch":
                                    continue
                                row = dict(
                                    N=n,
                                    B=b,
                                    dtype=dtype,
                                    backend=backend,
                                    scaling="adaptive"
                                    if backend == "torch"
                                    else args.scaling,
                                    scope=scope,
                                    operation=op,
                                    trial=trial,
                                )
                                if tuple(str(row[k]) for k in keys) in done:
                                    continue
                                try:
                                    start = time.perf_counter()
                                    fn()
                                    torch.cuda.synchronize()
                                    row["first_call_ms"] = (
                                        time.perf_counter() - start
                                    ) * 1000
                                    count = 32 if scope == "raw" and b <= 1024 else 1
                                    if scope == "raw":
                                        graph = torch.cuda.CUDAGraph()
                                        with torch.cuda.graph(graph):
                                            for _ in range(count):
                                                fn()
                                        fn = graph.replay
                                    samples = (
                                        np.asarray(
                                            triton.testing.do_bench(
                                                fn,
                                                warmup=20,
                                                rep=args.rep_ms,
                                                return_mode="all",
                                            )
                                        )
                                        / count
                                    )
                                    row.update(
                                        median_ms=float(np.median(samples)),
                                        mean_ms=float(np.mean(samples)),
                                        p99_ms=float(np.quantile(samples, 0.99)),
                                        variance_ms2=float(np.var(samples)),
                                        samples=len(samples),
                                        error="",
                                    )
                                except Exception as exc:
                                    row["error"] = repr(exc)
                                    failures.append(row)
                                writer.writerow(row)
                                f.flush()
                    print("completed", n, b, dtype, flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} benchmark cases failed; see {args.output}")


if __name__ == "__main__":
    main()
