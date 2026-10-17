#!/usr/bin/env bash
set -euo pipefail

# Activate the custom PyTorch environment before running this script.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
run_dir=${1:?Usage: run_timings.sh OUTPUT_DIRECTORY TORCH_LINALG_PYTHON}
linalg_python=${2:?Pass the stock PyTorch Python executable as the second argument}
shift 2
small_python=$(command -v python)
mkdir -p "$run_dir"
run_dir=$(cd "$run_dir" && pwd)
export PYTHONPATH="$(dirname "$script_dir")${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$run_dir"
rm -f "$run_dir/timing_run.exit"
rm -f "$run_dir/accuracy/figure_manifest.json"

stage() {
    local name=$1
    shift
    date -u
    echo "Starting $name"
    rm -f "$run_dir/$name.exit"
    local result=0
    "$@" > "$run_dir/$name.log" 2>&1 || result=$?
    echo "$result" > "$run_dir/$name.exit"
    echo "$name exit $result"
    return "$result"
}
trap 'result=$?; echo "$result" > "$run_dir/timing_run.exit"; exit "$result"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

for trial in 0 1 2; do
    backends=(torch_linalg torch_small_n)
    if (( trial % 2 == 1 )); then
        backends=(torch_small_n torch_linalg)
    fi
    for backend in "${backends[@]}"; do
        if [[ $backend == torch_linalg ]]; then
            stage "timing_${backend}_${trial}" /usr/bin/env -u LD_LIBRARY_PATH \
                VIRTUAL_ENV="$(dirname "$(dirname "$linalg_python")")" \
                "$linalg_python" "$script_dir/benchmark_native.py" \
                --backend "$backend" --trial "$trial" --output "$run_dir/timings" "$@"
        else
            stage "timing_${backend}_${trial}" "$small_python" \
                "$script_dir/benchmark_native.py" \
                --backend "$backend" --trial "$trial" --output "$run_dir/timings" "$@"
        fi
    done
done
stage summary "$small_python" "$script_dir/summarize_native.py" --output "$run_dir"
stage accuracy_reference /usr/bin/env -u LD_LIBRARY_PATH \
    VIRTUAL_ENV="$(dirname "$(dirname "$linalg_python")")" \
    "$linalg_python" "$script_dir/measure_accuracy.py" reference --output "$run_dir"
stage accuracy_compare "$small_python" "$script_dir/measure_accuracy.py" compare --output "$run_dir"
stage accuracy_plots "$small_python" "$script_dir/plot_accuracy.py" --output "$run_dir"
