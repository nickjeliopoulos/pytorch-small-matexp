import contextlib
import csv
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import summarize_native
from timing_metadata import (
    check_environment,
    check_shared_settings,
    load_timing_metadata,
    SHARED_SETTINGS,
)

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestTimingMetadata(TestCase):
    def _environment(self):
        return dict(
            gpu="A100",
            gpu_capability=[8, 0],
            gpu_memory_bytes=80_000_000_000,
            samples=2,
            warmup_calls=5,
            input="shared inputs",
            scope="wall",
            backward="graph prepared beforehand",
            num_threads=2,
            matmul_precision="highest",
            allow_tf32=False,
            grid=dict(n=[4], dtype=["fp32"], batch=[1], operation=["fwd", "bwd"]),
            torch="test-build",
            git="revision",
            library_sha256={"libtorch_cpu.so": "a"},
            started_at_utc="2026-01-01T00:00:00+00:00",
            completed_at_utc="2026-01-01T00:01:00+00:00",
        )

    def test_trials_and_hosts_can_differ(self):
        first = dict(trial=0, host="first", library_sha256={"libtorch_cpu.so": "a"})
        second = dict(first, trial=1, host="second")
        check_environment(first, second)

    @parametrize("setting", SHARED_SETTINGS)
    def test_reject_cross_backend_measurement_mismatch(self, setting):
        first = self._environment()
        second = dict(first, backend="torch_small_n", torch="different build")
        check_shared_settings(first, second)
        second[setting] = "changed"
        with self.assertRaisesRegex(
            RuntimeError, f"setting differs or is missing: {setting}"
        ):
            check_shared_settings(first, second)

    def test_summary_manifest_binds_artifacts_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            timings = output / "timings"
            timings.mkdir()
            for backend in ("torch_linalg", "torch_small_n"):
                for trial in range(3):
                    path = timings / f"{backend}_trial{trial}.csv"
                    metadata = dict(self._environment(), backend=backend, trial=trial)
                    path.with_suffix(".json").write_text(json.dumps(metadata))
                    rows = [
                        dict(
                            n=4,
                            dtype="fp32",
                            batch=1,
                            operation=op,
                            backend=backend,
                            trial=trial,
                            samples=2,
                            input_sha256="input",
                            upstream_sha256="grad",
                            median_ms=1,
                        )
                        for op in ("fwd", "bwd")
                    ]
                    with path.open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
            with (
                mock.patch.object(
                    sys, "argv", ["summarize_native.py", "--output", directory]
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                summarize_native.main()
            manifest = json.loads((output / "wall_manifest.json").read_text())
            self.assertEqual(manifest["measurement_settings"]["gpu"], "A100")
            self.assertEqual(len(manifest["trials"]), 6)
            self.assertTrue(manifest["generated_at_utc"])
            for artifact in manifest["artifacts"]:
                self.assertEqual(
                    hashlib.sha256(
                        (output / artifact["path"]).read_bytes()
                    ).hexdigest(),
                    artifact["sha256"],
                )
            for path in timings.glob("torch_small_n*.json"):
                metadata = json.loads(path.read_text())
                metadata["gpu"] = "H100"
                path.write_text(json.dumps(metadata))
            with (
                mock.patch.object(
                    sys, "argv", ["summarize_native.py", "--output", directory]
                ),
                self.assertRaisesRegex(
                    RuntimeError, "setting differs or is missing: gpu"
                ),
            ):
                summarize_native.main()
            self.assertFalse((output / "wall_manifest.json").exists())

    @parametrize("library", ["libtorch_cpu.so", "libtorch_cuda.so"])
    def test_reject_changed_build(self, library):
        first = dict(trial=0, library_sha256={library: "before"})
        second = dict(trial=1, library_sha256={library: "after"})
        with self.assertRaisesRegex(RuntimeError, "environment changed"):
            check_environment(first, second)

    def test_csv_requires_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torch_small_n_trial0.csv"
            path.write_text("n,batch\n4,1\n")
            with self.assertRaisesRegex(RuntimeError, "no environment metadata"):
                load_timing_metadata(path)
            metadata = dict(backend="torch_small_n", trial=0)
            path.with_suffix(".json").write_text(json.dumps(metadata))
            self.assertEqual(load_timing_metadata(path), metadata)


if __name__ == "__main__":
    run_tests()
