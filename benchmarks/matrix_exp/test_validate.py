import contextlib
import csv
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

import validate

import torch
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMatrixExpValidation(TestCase):
    def setUp(self):
        super().setUp()
        # Report tests do not need the command's process-wide tuning settings.
        for name in ("set_float32_matmul_precision", "set_num_threads"):
            patcher = mock.patch.object(torch, name)
            patcher.start()
            self.addCleanup(patcher.stop)

    @onlyCUDA
    def test_reports_are_replaced(self, device):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "correctness.csv"
            argv = [
                "validate.py",
                "--n",
                "4",
                "--dtype",
                "fp32",
                "--backends",
                "torch",
                "--operations",
                "fwd",
                "--batch",
                "2",
                "--output",
                str(path),
            ]
            row_counts = None
            for fail in (False, True, False):

                def result(a, backend, scaling, fail=fail):
                    return (
                        torch.full_like(a, float("nan"))
                        if fail
                        else validate.reference(a).to(a.dtype)
                    )

                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch.object(validate, "expm", side_effect=result),
                    contextlib.redirect_stdout(io.StringIO()),
                    self.assertRaises(SystemExit) as raised,
                ):
                    validate.main()
                self.assertEqual(raised.exception.code, int(fail))
                counts = []
                for report in (path, path.with_name("correctness_per_matrix.csv")):
                    with report.open() as stream:
                        rows = list(csv.DictReader(stream))
                    self.assertTrue(rows)
                    self.assertTrue(all(row["passed"] == str(not fail) for row in rows))
                    counts.append(len(rows))
                if row_counts is None:
                    row_counts = counts
                self.assertEqual(counts, row_counts)
                summary = json.loads(path.with_suffix(".summary.json").read_text())
                self.assertEqual(summary["passed"], not fail)
                self.assertEqual(bool(summary["failures"]), fail)

    @onlyCUDA
    def test_interruption_invalidates_summary(self, device):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "correctness.csv"
            summary = path.with_suffix(".summary.json")
            summary.write_text(json.dumps({"passed": True, "failures": []}))
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "validate.py",
                        "--n",
                        "4",
                        "--dtype",
                        "fp32",
                        "--batch",
                        "2",
                        "--operations",
                        "fwd",
                        "--output",
                        str(path),
                    ],
                ),
                mock.patch.object(
                    validate, "expm", side_effect=RuntimeError("interrupted")
                ),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                validate.main()
            self.assertFalse(summary.exists())

    def test_resume_is_rejected(self, device):
        with (
            mock.patch.object(sys, "argv", ["validate.py", "--resume"]),
            mock.patch.object(
                validate,
                "make_input",
                side_effect=AssertionError("--resume was accepted"),
            ),
            contextlib.redirect_stderr(io.StringIO()) as stderr,
            self.assertRaises(SystemExit) as raised,
        ):
            validate.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("unrecognized arguments: --resume", stderr.getvalue())


instantiate_device_type_tests(TestMatrixExpValidation, globals(), only_for="cuda")

if __name__ == "__main__":
    run_tests()
