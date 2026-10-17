"""Build provenance checks shared by the timing runner and summarizer."""

import json


def load_timing_metadata(csv_path):
    path = csv_path.with_suffix(".json")
    if not path.exists():
        raise RuntimeError(f"Timing CSV has no environment metadata: {csv_path}")
    return json.loads(path.read_text())


def check_environment(previous, current):
    # Trials may run on different allocated hosts, but must use the same build,
    # GPU model, inputs, and timing configuration for each backend.
    def fingerprint(metadata):
        return {k: v for k, v in metadata.items() if k not in ("trial", "host")}

    if fingerprint(previous) != fingerprint(current):
        raise RuntimeError("Timing environment changed; choose a new output directory")
