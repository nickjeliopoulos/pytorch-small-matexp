"""Fresh per-matrix comparisons against the separately installed stock PyTorch."""

import argparse
import csv
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import torch


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
NORM_GRID = [0.01, 0.1, 1.0, 3.0, 6.0, 10.0]


def tensor_hash(x):
    return hashlib.sha256(
        x.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def load_manifest(output):
    manifest = json.loads((output / "wall_manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        with (output / artifact["path"]).open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != artifact["sha256"]:
            raise RuntimeError(f"Timing artifact changed: {artifact['path']}")
    return manifest


def environment(backend, manifest):
    previous = next(
        x["environment"]
        for x in manifest["trials"]
        if x["environment"]["backend"] == backend
    )
    hashes = {}
    for name in ("libtorch_cpu.so", "libtorch_cuda.so"):
        path = Path(torch._C.__file__).parent / "lib" / name
        with path.open("rb") as f:
            hashes[str(path)] = hashlib.file_digest(f, "sha256").hexdigest()
    if hashes != previous["library_sha256"]:
        raise RuntimeError(f"{backend}: binaries no longer match the saved timing run")
    properties = torch.cuda.get_device_properties(0)
    if (
        properties.name != previous["gpu"]
        or [properties.major, properties.minor] != previous["gpu_capability"]
        or properties.total_memory != previous["gpu_memory_bytes"]
    ):
        raise RuntimeError("GPU differs from the saved timing run")
    return dict(
        backend=backend,
        torch=torch.__version__,
        torch_file=torch.__file__,
        git=torch.version.git_version,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        host=platform.node(),
        library_sha256=hashes,
        utc=datetime.now(timezone.utc).isoformat(),
    )


def specs(grid, matrices=129, timing_batch=None):
    if matrices < 1:
        raise ValueError("The accuracy matrix count must be positive")
    if timing_batch is None:
        timing_batch = 8192 if 8192 in grid["batch"] else max(grid["batch"])
    if timing_batch not in grid["batch"]:
        raise ValueError("Accuracy timing batch must occur in the timing grid")
    if any(n not in (4, 8, 16, 32) for n in grid["n"]):
        raise ValueError("Accuracy figures support N=4,8,16,32")
    result = []
    for n in grid["n"]:
        for dtype in grid["dtype"]:
            for family in ["random", "skew", "neg_psd", "diagonal"]:
                for norm in NORM_GRID:
                    result.append(
                        dict(
                            suite="core",
                            n=n,
                            dtype=dtype,
                            batch=matrices,
                            family=family,
                            target_norm=norm,
                        )
                    )
            for family in ["skew", "neg_psd", "rotation"]:
                for norm in [30.0, 100.0, 1000.0]:
                    result.append(
                        dict(
                            suite="stress",
                            n=n,
                            dtype=dtype,
                            batch=matrices,
                            family=family,
                            target_norm=norm,
                        )
                    )
            for family, norm in [("zero", 0.0), ("identity", 1.0)]:
                result.append(
                    dict(
                        suite="analytic",
                        n=n,
                        dtype=dtype,
                        batch=17,
                        family=family,
                        target_norm=norm,
                    )
                )
            result.append(
                dict(
                    suite="timing",
                    n=n,
                    dtype=dtype,
                    batch=timing_batch,
                    family="random",
                    target_norm=3.0,
                )
            )
    return result


def make_inputs(spec, output):
    n, b, family = spec["n"], spec["batch"], spec["family"]
    if spec["suite"] == "timing":
        saved = torch.load(
            output / "timings/inputs" / f"{n}_{spec['dtype']}_{b}.pt", weights_only=True
        )
        return saved["a"], saved["g"]
    gen = torch.Generator().manual_seed(841 + n)
    x = torch.randn(b, n, n, generator=gen, dtype=torch.float64)
    if family == "skew":
        x = x - x.mT
    elif family == "neg_psd":
        x = -(x @ x.mT)
    elif family == "diagonal":
        x = torch.diag_embed(x.diagonal(dim1=-2, dim2=-1))
    elif family == "rotation":
        x.zero_()
        i = torch.arange(b)
        p, q = i % n, (i + 1) % n
        x[i, p, q], x[i, q, p] = -1, 1
    elif family == "zero":
        x.zero_()
    elif family == "identity":
        x = torch.eye(n, dtype=torch.float64).expand(b, n, n).clone()
    elif family != "random":
        raise ValueError(family)
    norms = torch.linalg.matrix_norm(x, ord=1).clamp_min(1e-300)
    x = x * (spec["target_norm"] / norms).view(-1, 1, 1)
    gen.manual_seed(846 + n)
    g = torch.randn(b, n, n, generator=gen, dtype=torch.float64) * 0.2
    return x.to(DTYPES[spec["dtype"]]), g.to(DTYPES[spec["dtype"]])


def calculate(a, g, backend):
    a = a.cuda().requires_grad_()
    g = g.cuda()
    if backend == "torch_linalg":
        y = torch.linalg.matrix_exp(a.float()).to(a.dtype)
    else:
        y = torch.linalg.matrix_exp(a)
    (dx,) = torch.autograd.grad(y, a, g)
    return {"fwd": y.detach().cpu(), "bwd": dx.detach().cpu()}


def matrix_errors(value, expected):
    actual = value.double().flatten(1)
    ref = expected.double().flatten(1)
    actual_ok, ref_ok = actual.isfinite().all(1), ref.isfinite().all(1)
    finite = actual_ok & ref_ok
    delta = actual - ref
    ref_norm = torch.linalg.vector_norm(ref, dim=1)
    diff_norm = torch.linalg.vector_norm(delta, dim=1)
    relative = torch.where(
        ref_norm > 0,
        diff_norm / ref_norm,
        torch.where(diff_norm == 0, 0.0, float("inf")),
    )
    relative = torch.where(finite, relative, float("inf"))
    statuses = []
    for i in range(len(value)):
        if not actual_ok[i] and not ref_ok[i]:
            status = "both_nonfinite"
        elif not actual_ok[i]:
            status = "actual_nonfinite"
        elif not ref_ok[i]:
            status = "reference_nonfinite"
        elif not torch.isfinite(relative[i]):
            status = "zero_reference_nonzero_actual"
        else:
            status = "finite"
        statuses.append(status)
    max_abs = delta.abs().amax(1)
    return dict(
        relative=relative,
        max_abs=max_abs,
        reference_norm=ref_norm,
        normalized=max_abs / (1 + ref.abs().amax(1)),
        exact=(actual == ref).all(1) & finite,
        statuses=statuses,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["reference", "compare"])
    parser.add_argument(
        "--output", type=Path, required=True, help="Timing report directory"
    )
    parser.add_argument("--matrices", type=int, default=129)
    parser.add_argument("--timing-batch", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    root = output / "accuracy"
    root.mkdir(exist_ok=True)
    for path in [
        root / "figure_manifest.json",
        root / "case_summary.json",
        root / "torch_small_n_metadata.json",
    ]:
        path.unlink(missing_ok=True)
    for pattern in (
        "accuracy_overview_*.*",
        "cdf_core_*.*",
        "cdf_stress_*.*",
        "speedup_accuracy.*",
        "distribution_summary.json",
    ):
        for path in root.glob(pattern):
            path.unlink()
    if args.stage == "reference":
        (root / "torch_linalg_metadata.json").unlink(missing_ok=True)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    backend = "torch_linalg" if args.stage == "reference" else "torch_small_n"
    manifest = load_manifest(output)
    manifest_hash = hashlib.sha256(
        (output / "wall_manifest.json").read_bytes()
    ).hexdigest()
    metadata = environment(backend, manifest)
    all_specs = specs(
        manifest["measurement_settings"]["grid"], args.matrices, args.timing_batch
    )
    metadata.update(timing_manifest_sha256=manifest_hash, specs=all_specs)
    metadata.update(
        metric="||actual-reference||_F / ||reference||_F per matrix, evaluated in FP64",
        zero_reference="0 if both outputs zero, infinity otherwise",
        nonfinite="infinity for CDF accounting; explicit status in CSV",
        reference_policy="Stock FP32 matrix_exp then cast to input dtype, including autograd through casts",
        input_policy="CPU FP64 Gaussian-derived families normalized to target 1-norm, then cast; timing suite uses exact saved timing tensors",
        input_seeds="841+N for input, 846+N for upstream; timing suite original saved seeds",
        core_norms=NORM_GRID,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    if args.stage == "compare":
        reference_metadata = json.loads(
            (root / "torch_linalg_metadata.json").read_text()
        )
        if (
            reference_metadata.get("status") != "complete"
            or reference_metadata["timing_manifest_sha256"] != manifest_hash
            or reference_metadata["specs"] != all_specs
            or reference_metadata["script_sha256"] != metadata["script_sha256"]
        ):
            raise RuntimeError(
                "Stock accuracy collection is incomplete or belongs to another run"
            )
    (root / f"{backend}_metadata.json").write_text(json.dumps(metadata, indent=2))
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    if args.stage == "reference":
        for i, spec in enumerate(all_specs):
            a, g = make_inputs(spec, output)
            hashes = dict(input_sha256=tensor_hash(a), upstream_sha256=tensor_hash(g))
            if spec["suite"] == "timing":
                with (output / "timings/torch_linalg_trial0.csv").open() as f:
                    matched = [
                        r
                        for r in csv.DictReader(f)
                        if (int(r["n"]), r["dtype"], int(r["batch"]))
                        == (spec["n"], spec["dtype"], spec["batch"])
                    ]
                if len(matched) != 2 or any(
                    r[k] != v for r in matched for k, v in hashes.items()
                ):
                    raise RuntimeError("Timing input hashes do not match")
            result = calculate(a, g, backend)
            torch.save(
                dict(spec=spec, a=a, g=g, reference=result, **hashes),
                data_dir / f"case_{i:03d}.pt",
            )
            if i % 12 == 0:
                print(i, len(all_specs), spec, flush=True)
        metadata["cases"] = len(all_specs)
    else:
        worst_dir = root / "worst_cases"
        worst_dir.mkdir(exist_ok=True)
        fields = [
            "case_id",
            "suite",
            "n",
            "dtype",
            "batch",
            "family",
            "target_norm",
            "operation",
            "matrix_index",
            "input_norm1",
            "relative_fro_error",
            "normalized_max_error",
            "max_abs_error",
            "reference_fro_norm",
            "exact",
            "status",
        ]
        summaries = []
        with (root / "per_matrix_errors.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, spec in enumerate(all_specs):
                saved = torch.load(data_dir / f"case_{i:03d}.pt", weights_only=True)
                if (
                    saved["spec"] != spec
                    or tensor_hash(saved["a"]) != saved["input_sha256"]
                    or tensor_hash(saved["g"]) != saved["upstream_sha256"]
                ):
                    raise RuntimeError("Saved input mismatch")
                results = calculate(saved["a"], saved["g"], backend)
                norms = torch.linalg.matrix_norm(saved["a"].double(), ord=1).numpy()
                for operation, value in results.items():
                    measured = matrix_errors(value, saved["reference"][operation])
                    rel, exact = measured["relative"], measured["exact"]
                    statuses = measured["statuses"]
                    for j in range(spec["batch"]):
                        writer.writerow(
                            dict(
                                case_id=i,
                                **spec,
                                operation=operation,
                                matrix_index=j,
                                input_norm1=float(norms[j]),
                                relative_fro_error=rel[j].item(),
                                normalized_max_error=measured["normalized"][j].item(),
                                max_abs_error=measured["max_abs"][j].item(),
                                reference_fro_norm=measured["reference_norm"][j].item(),
                                exact=bool(exact[j]),
                                status=statuses[j],
                            )
                        )
                    finite_rel = rel[rel.isfinite()].numpy()
                    summary = dict(
                        case_id=i,
                        **spec,
                        operation=operation,
                        input_sha256=saved["input_sha256"],
                        upstream_sha256=saved["upstream_sha256"],
                        matrices=len(rel),
                        exact_matrices=int(exact.sum()),
                        unplottable_matrices=int((~rel.isfinite()).sum()),
                        status_counts={
                            s: statuses.count(s) for s in sorted(set(statuses))
                        },
                        max_relative_error=rel.max().item(),
                        finite_median=np.median(finite_rel).item()
                        if len(finite_rel)
                        else None,
                        finite_p95=np.quantile(finite_rel, 0.95).item()
                        if len(finite_rel)
                        else None,
                        finite_p99=np.quantile(finite_rel, 0.99).item()
                        if len(finite_rel)
                        else None,
                    )
                    summaries.append(
                        {
                            key: str(v)
                            if isinstance(v, float) and not math.isfinite(v)
                            else v
                            for key, v in summary.items()
                        }
                    )
                    worst = int(rel.argmax())
                    torch.save(
                        dict(
                            spec=spec,
                            operation=operation,
                            index=worst,
                            input=saved["a"][worst],
                            upstream=saved["g"][worst],
                            actual=value[worst],
                            reference=saved["reference"][operation][worst],
                        ),
                        worst_dir / f"case_{i:03d}_{operation}.pt",
                    )
                f.flush()
                if i % 12 == 0:
                    print(i, len(all_specs), spec, flush=True)
        (root / "case_summary.json").write_text(
            json.dumps(summaries, indent=2, allow_nan=False)
        )
        metadata["cases"] = len(all_specs)
        metadata["comparisons"] = len(summaries)
    metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["status"] = "complete"
    (root / f"{backend}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print("complete", backend, metadata["cases"], flush=True)


if __name__ == "__main__":
    main()
