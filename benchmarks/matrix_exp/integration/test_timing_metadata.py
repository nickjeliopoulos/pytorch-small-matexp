import json
import tempfile
from pathlib import Path

from timing_metadata import check_environment, load_timing_metadata

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestTimingMetadata(TestCase):
    def test_trials_and_hosts_can_differ(self):
        first = dict(trial=0, host="first", library_sha256={"libtorch_cpu.so": "a"})
        second = dict(first, trial=1, host="second")
        check_environment(first, second)

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
