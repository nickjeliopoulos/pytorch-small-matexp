"""Public-API wall timings, one installed PyTorch/backend per process."""

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from matrix_exp_bench.cases import DTYPE_MAP, make_input
from timing_metadata import (
    check_environment,
    check_shared_settings,
    load_timing_metadata,
    source_identity,
)

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("torch_small_n", "torch_linalg"), required=True
    )
    parser.add_argument("--trial", type=int, required=True)
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
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=True)
    input_dir = args.output / "inputs"
    input_dir.mkdir(exist_ok=True)
    # The manual derivative lives in libtorch_cpu even for CUDA tensors.
    # Hash both libraries: an editable build can change either without changing
    # torch.version.git_version, and resumed timings must not mix those builds.
    library_dir = Path(torch._C.__file__).parent / "lib"
    library_sha256 = {}
    for name in ("libtorch_cpu.so", "libtorch_cuda.so"):
        path = library_dir / name
        with path.open("rb") as f:
            library_sha256[str(path)] = hashlib.file_digest(f, "sha256").hexdigest()
    metadata = dict(
        backend=args.backend,
        trial=args.trial,
        samples=args.samples,
        warmup_calls=5,
        torch=torch.__version__,
        git=torch.version.git_version,
        torch_file=torch.__file__,
        cuda=torch.version.cuda,
        library_sha256=library_sha256,
        gpu=torch.cuda.get_device_name(),
        gpu_capability=list(torch.cuda.get_device_capability()),
        gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        host=platform.node(),
        num_threads=torch.get_num_threads(),
        matmul_precision=torch.get_float32_matmul_precision(),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        grid=dict(
            n=args.n, dtype=args.dtype, batch=args.batch, operation=["fwd", "bwd"]
        ),
        runner_source=source_identity(Path(__file__).parent),
        torch_checkout=source_identity(Path(torch.__file__).parent),
        input="seed 841, prescribed 1-norm 3; upstream seed 846, normal times 0.2",
        scope="perf_counter + CUDA synchronize; allocations, conversions, dispatch, autograd included",
        backward="autograd.grad(retain_graph=True); forward graph built outside timing",
        config=torch.__config__.show(),
    )
    stem = f"{args.backend}_trial{args.trial}"
    meta_path = args.output / f"{stem}.json"
    path = args.output / f"{stem}.csv"
    if path.exists():
        check_environment(load_timing_metadata(path), metadata)
    for previous_path in args.output.glob(f"{args.backend}_trial*.json"):
        check_environment(json.loads(previous_path.read_text()), metadata)
    for previous_path in args.output.glob("*_trial*.json"):
        check_shared_settings(json.loads(previous_path.read_text()), metadata)
    now = datetime.now(timezone.utc).isoformat()
    previous_metadata = load_timing_metadata(path) if path.exists() else {}
    metadata["started_at_utc"] = previous_metadata.get("started_at_utc", now)
    metadata["invocations"] = previous_metadata.get("invocations", []) + [
        dict(
            started_at_utc=now,
            host=platform.node(),
            command=[sys.executable, *sys.argv],
        )
    ]
    (args.output.parent / "wall_manifest.json").unlink(missing_ok=True)
    with (args.output / "runs.jsonl").open("a") as f:
        f.write(json.dumps(dict(timestamp=time.time(), **metadata)) + "\n")
    meta_path.write_text(json.dumps(metadata, indent=2))
    previous = list(csv.DictReader(path.open())) if path.exists() else []
    complete = {
        (int(r["n"]), r["dtype"], int(r["batch"]), r["operation"]) for r in previous
    }
    fields = [
        "n",
        "dtype",
        "batch",
        "operation",
        "trial",
        "backend",
        "host",
        "policy",
        "input_sha256",
        "upstream_sha256",
        "median_ms",
        "mean_ms",
        "p99_ms",
        "variance_ms2",
        "samples",
        "raw_ms",
    ]
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not previous:
            writer.writeheader()
        for dtype_name in args.dtype:
            for n in args.n:
                for batch in args.batch:
                    if all(
                        (n, dtype_name, batch, op) in complete for op in ("fwd", "bwd")
                    ):
                        continue
                    input_path = input_dir / f"{n}_{dtype_name}_{batch}.pt"
                    if input_path.exists():
                        saved = torch.load(
                            input_path, map_location="cuda", weights_only=True
                        )
                        if saved["spec"] != metadata["input"]:
                            raise RuntimeError(
                                "Cached benchmark input configuration changed"
                            )
                        a, g = saved["a"], saved["g"]
                        del saved
                    else:
                        if args.backend != "torch_linalg":
                            raise RuntimeError(
                                "Run torch_linalg trial 0 first to create canonical inputs"
                            )
                        a = make_input(
                            batch, n, DTYPE_MAP[dtype_name], "large", 3.0, seed=841
                        )
                        generator = torch.Generator(device="cuda").manual_seed(846)
                        g = (
                            torch.randn(
                                a.shape,
                                dtype=a.dtype,
                                device=a.device,
                                generator=generator,
                            )
                            * 0.2
                        )
                        temporary = input_path.with_suffix(".tmp")
                        torch.save(
                            dict(
                                a=a.cpu(),
                                g=g.cpu(),
                                spec=metadata["input"],
                                created_with=str(torch.__version__),
                            ),
                            temporary,
                        )
                        temporary.replace(input_path)
                    input_hash = hashlib.sha256(
                        a.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                    ).hexdigest()
                    upstream_hash = hashlib.sha256(
                        g.contiguous().view(torch.uint8).cpu().numpy().tobytes()
                    ).hexdigest()
                    if args.backend == "torch_small_n":
                        forward = torch.linalg.matrix_exp
                        policy = "input_dtype"
                    else:

                        def forward(x):
                            return torch.linalg.matrix_exp(x.float()).to(x.dtype)

                        policy = (
                            "fp32"
                            if a.dtype == torch.float32
                            else "fp32_with_input_output_casts"
                        )
                    for operation in ("fwd", "bwd"):
                        if (n, dtype_name, batch, operation) in complete:
                            continue
                        if operation == "fwd":

                            def call():
                                return forward(a)
                        else:
                            x = a.detach().requires_grad_()
                            y = forward(x)

                            def call():
                                return torch.autograd.grad(y, x, g, retain_graph=True)

                        for _ in range(5):
                            call()
                        torch.cuda.synchronize()
                        samples = []
                        for _ in range(args.samples):
                            start = time.perf_counter()
                            call()
                            torch.cuda.synchronize()
                            samples.append((time.perf_counter() - start) * 1000)
                        writer.writerow(
                            dict(
                                n=n,
                                dtype=dtype_name,
                                batch=batch,
                                operation=operation,
                                trial=args.trial,
                                backend=args.backend,
                                host=platform.node(),
                                policy=policy,
                                input_sha256=input_hash,
                                upstream_sha256=upstream_hash,
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
                            del x, y
                        print(
                            args.backend,
                            args.trial,
                            dtype_name,
                            n,
                            batch,
                            operation,
                            flush=True,
                        )
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
