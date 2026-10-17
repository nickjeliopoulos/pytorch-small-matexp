"""Descriptive numerical reports; no accuracy acceptance thresholds."""

import json
import math


def write_error_summary(path, rows, error_key):
    ranked = sorted(rows, key=lambda row: row[error_key], reverse=True)
    nonfinite = [row for row in rows if not row["finite"]]
    summary = dict(
        status="complete",
        comparisons=len(rows),
        interpretation="Descriptive errors, not a numerical acceptance verdict",
        ranking_metric=error_key,
        nonfinite_comparisons=len(nonfinite),
        nonfinite_cases=nonfinite,
        largest_errors=ranked[:20],
    )
    # Preserve nonfinite measurements explicitly in standards-compliant JSON.
    for key in ("nonfinite_cases", "largest_errors"):
        summary[key] = [
            {
                name: str(value)
                if isinstance(value, float) and not math.isfinite(value)
                else value
                for name, value in row.items()
            }
            for row in summary[key]
        ]
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
