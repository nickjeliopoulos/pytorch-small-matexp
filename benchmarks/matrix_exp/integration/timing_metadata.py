"""Build provenance checks shared by the timing runner and summarizer."""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


SHARED_SETTINGS = (
    "gpu",
    "gpu_capability",
    "gpu_memory_bytes",
    "samples",
    "warmup_calls",
    "input",
    "scope",
    "backward",
    "num_threads",
    "matmul_precision",
    "allow_tf32",
    "grid",
)


def source_identity(directory):
    directory = Path(directory).resolve()
    try:
        root = Path(
            subprocess.check_output(
                ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except subprocess.CalledProcessError:
        return None
    paths = [
        "aten/src/ATen/native/SmallMatrixExp.h",
        "aten/src/ATen/native/LinearAlgebra.cpp",
        "aten/src/ATen/native/cuda/MatrixExpKernel.cu",
        "aten/src/ATen/native/cuda/MatrixExpSmall.cuh",
        "torch/csrc/autograd/FunctionsManual.cpp",
        "test/test_linalg.py",
        "benchmarks/matrix_exp",
    ]
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    patch = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--binary", "HEAD", "--", *paths],
    )
    files = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(root),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
                "--",
                *paths,
            ],
        )
        .decode()
        .split("\0")
    )
    hashes = {}
    for name in sorted(set(files) - {""}):
        path = root / name
        hashes[name] = (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        )
    return dict(
        root=str(root),
        git_revision=revision,
        tracked_patch_sha256=hashlib.sha256(patch).hexdigest(),
        has_tracked_changes=bool(patch),
        source_sha256=hashes,
    )


def load_timing_metadata(csv_path):
    path = csv_path.with_suffix(".json")
    if not path.exists():
        raise RuntimeError(f"Timing CSV has no environment metadata: {csv_path}")
    return json.loads(path.read_text())


def check_environment(previous, current):
    # Trials may run on different allocated hosts, but must use the same build,
    # GPU model, inputs, and timing configuration for each backend.
    def fingerprint(metadata):
        return {
            k: v
            for k, v in metadata.items()
            if k
            not in (
                "trial",
                "host",
                "started_at_utc",
                "completed_at_utc",
                "invocations",
            )
        }

    if fingerprint(previous) != fingerprint(current):
        raise RuntimeError("Timing environment changed; choose a new output directory")


def check_shared_settings(first, second):
    for key in SHARED_SETTINGS:
        if key not in first or key not in second or first[key] != second[key]:
            raise RuntimeError(
                f"Timing measurement setting differs or is missing: {key}"
            )


def write_manifest(output, trials, artifacts):
    records = []
    for path in artifacts:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append(dict(path=str(path.relative_to(output)), sha256=digest))
    manifest = dict(
        schema_version=1,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        kind="installed_public_api_wall_timings",
        measurement_settings={
            key: trials[0]["environment"][key] for key in SHARED_SETTINGS
        },
        trials=trials,
        report_source=source_identity(Path(__file__).parent),
        artifacts=records,
        identity_note="Checkout hashes describe source present at measurement time; shared-library hashes identify loaded binaries. A base Git revision alone does not identify an editable build.",
    )
    temporary = output / "wall_manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "wall_manifest.json")
