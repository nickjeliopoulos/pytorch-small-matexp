import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
from measure_accuracy import load_manifest, matrix_errors, specs
from plot_accuracy import accuracy_points, cdf, load_measurements
import validate_pytorch

import torch
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestAccuracyReporting(TestCase):
    @parametrize("probe_state", ["missing", "invalid"])
    def test_failed_probe_load_invalidates_summary(self, probe_state):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "numerical_summary.json"
            summary.write_text(json.dumps({"status": "complete"}))
            library = root / "probe.so"
            if probe_state == "invalid":
                library.write_text("invalid shared library")
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "validate_pytorch.py",
                        "--library",
                        str(library),
                        "--output",
                        str(root),
                    ],
                ),
                mock.patch.object(torch, "set_num_threads"),
                mock.patch.object(torch, "set_float32_matmul_precision"),
                self.assertRaisesRegex(
                    OSError, r"Could not load this library: .*probe\.so"
                ),
            ):
                validate_pytorch.main()
            self.assertFalse(summary.exists())

    def test_relative_error_is_per_matrix(self):
        reference = torch.tensor([[[3.0, 4.0]], [[0.003, 0.004]]])
        actual = reference * 2
        measured = matrix_errors(actual, reference)
        self.assertEqual(measured["relative"], torch.ones(2, dtype=torch.float64))

    def test_zero_and_nonfinite_reference(self):
        actual = torch.tensor([0.0, 1.0, float("inf"), 1.0, float("nan")]).view(
            -1, 1, 1
        )
        reference = torch.tensor([0.0, 0.0, 1.0, float("inf"), float("nan")]).view(
            -1, 1, 1
        )
        measured = matrix_errors(actual, reference)
        self.assertEqual(
            measured["relative"],
            torch.tensor(
                [0.0, float("inf"), float("inf"), float("inf"), float("inf")],
                dtype=torch.float64,
            ),
        )
        self.assertEqual(
            measured["statuses"],
            [
                "finite",
                "zero_reference_nonzero_actual",
                "actual_nonfinite",
                "reference_nonfinite",
                "both_nonfinite",
            ],
        )
        self.assertEqual(
            measured["exact"], torch.tensor([True, False, False, False, False])
        )

    def test_cdf_retains_zero_small_positive_and_nonfinite_samples(self):
        fig, ax = plt.subplots()
        try:
            cdf(ax, [0.0, 1e-14, 1e-3, float("inf")], "error", "blue", 0.1)
            x, y = ax.lines[0].get_data()
            self.assertIn(1e-14, x)
            self.assertEqual(y[-1], 0.75)
            self.assertEqual(y[np.flatnonzero(x == 0)[-1]], 0.25)
        finally:
            plt.close(fig)

    def test_subset_uses_a_measured_batch(self):
        grid = dict(n=[4], dtype=["fp32"], batch=[1, 32])
        cases = specs(grid, matrices=3)
        self.assertEqual(len(cases), 36)
        self.assertEqual(cases[-1]["batch"], 32)
        self.assertEqual(cases[0]["batch"], 3)
        with self.assertRaisesRegex(ValueError, "must occur in the timing grid"):
            specs(grid, timing_batch=8192)

    def _write_accuracy(self, root):
        spec = dict(
            suite="core", n=4, dtype="fp32", batch=2, family="random", target_norm=1.0
        )
        metadata = dict(
            status="complete",
            timing_manifest_sha256="timing",
            specs=[spec],
            script_sha256="script",
        )
        for backend in ["torch_linalg", "torch_small_n"]:
            (root / f"{backend}_metadata.json").write_text(json.dumps(metadata))
        summaries = [
            dict(
                case_id=0,
                operation=op,
                **spec,
                max_relative_error=0.1,
                exact_matrices=1,
                unplottable_matrices=0,
            )
            for op in ["fwd", "bwd"]
        ]
        (root / "case_summary.json").write_text(json.dumps(summaries))
        rows = [
            dict(case_id=0, operation=op, matrix_index=i, relative_fro_error=value)
            for op in ["fwd", "bwd"]
            for i, value in enumerate([0.0, 0.1])
        ]
        with (root / "per_matrix_errors.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @parametrize("failure", ["stale", "incomplete", "truncated", "duplicate"])
    def test_reject_incomplete_or_stale_numerical_results(self, failure):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_accuracy(root)
            _, summaries, grouped = load_measurements(root, "timing")
            self.assertEqual(len(summaries), 2)
            self.assertEqual(len(grouped), 2)
            if failure in ["stale", "incomplete"]:
                path = root / "torch_small_n_metadata.json"
                metadata = json.loads(path.read_text())
                metadata[
                    "timing_manifest_sha256" if failure == "stale" else "status"
                ] = "changed"
                path.write_text(json.dumps(metadata))
            else:
                path = root / "per_matrix_errors.csv"
                lines = path.read_text().splitlines()
                lines = lines[:-1] if failure == "truncated" else lines + [lines[-1]]
                path.write_text("\n".join(lines) + "\n")
            with self.assertRaises(RuntimeError):
                load_measurements(root, "timing")

    def test_timing_manifest_rejects_modified_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "wall_summary.csv"
            path.write_text("original\n")
            record = dict(
                path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            (root / "wall_manifest.json").write_text(
                json.dumps(dict(artifacts=[record]))
            )
            load_manifest(root)
            path.write_text("modified\n")
            with self.assertRaisesRegex(RuntimeError, "Timing artifact changed"):
                load_manifest(root)

    def test_scatter_requires_identical_inputs_in_every_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = dict(
                suite="timing",
                n=4,
                dtype="fp32",
                batch=2,
                operation="fwd",
                family="random",
                target_norm=3.0,
                matrices=2,
                exact_matrices=1,
                unplottable_matrices=0,
                max_relative_error=0.01,
                finite_median=0.005,
                finite_p95=0.0095,
                finite_p99=0.0099,
                input_sha256="input",
                upstream_sha256="upstream",
            )
            trials = []
            for backend, median in [("torch_linalg", 2), ("torch_small_n", 1)]:
                for trial in range(3):
                    path = root / f"{backend}_{trial}.csv"
                    row = dict(
                        n=4,
                        dtype="fp32",
                        batch=2,
                        operation="fwd",
                        backend=backend,
                        trial=trial,
                        median_ms=median,
                        input_sha256="input",
                        upstream_sha256="upstream",
                    )
                    with path.open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)
                    trials.append(dict(csv=path.name))
            (root / "wall_summary.csv").write_text(
                "n,dtype,batch,operation,torch_linalg_median_ms,torch_small_n_median_ms,speedup\n4,fp32,2,fwd,2,1,2\n"
            )
            points = accuracy_points(root, dict(trials=trials), [case])
            self.assertEqual(points[0]["speedup"], 2)
            path.write_text(path.read_text().replace("upstream\n", "different\n"))
            with self.assertRaisesRegex(
                RuntimeError, "Accuracy and timing inputs differ"
            ):
                accuracy_points(root, dict(trials=trials), [case])


if __name__ == "__main__":
    run_tests()
