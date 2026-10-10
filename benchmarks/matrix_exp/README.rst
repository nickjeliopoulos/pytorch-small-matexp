Small-matrix exponential checkpoint
===================================

This directory preserves the standalone CUDA design measured before ATen
integration. It builds against an installed PyTorch release and does not
override ``torch.linalg.matrix_exp``. No files under ``agent_space`` are
required to build or run it.

The CUDA sources are copied from the validated experiment, with local variable
names adjusted for repository lint. Python sources have repository formatting.
``provenance.json`` records original and checkpoint hashes. The measured batch-65,536 summary,
Nsight counters, and launch selections are retained in ``baseline_results``;
those measurements describe the original experiment binaries, not a claim that
every rebuild will reproduce identical timings.

Backends and precision
----------------------

``scalar`` uses register rows and subgroup shuffles at N=4/8, and cooperative
FP32 scalar-FMA tiles at N=16/32. ``mma`` retains the row kernels at N=4/8 and
uses BF16 tensor-core operands with FP32 accumulation at N=16/32. The latter
uses BF16 products for both BF16 and FP16 outputs in this historical checkpoint;
an explicitly requested FP32 output also retains this approximate arithmetic.
BF16 intermediates for FP16 inputs are permitted by the experimental precision
policy. All backends preserve output shape, dtype, and device.

``triton_original`` is the preserved hyperconnections implementation, using
the structured backward recurrence. Its N>=16 low-precision path also uses
BF16 products. ``torch`` promotes inputs to FP32, calls stock PyTorch, and casts
back; wrapper timing includes those conversions.

CUDA supports fixed scaling (default S=2) and per-matrix adaptive scaling,
including the augmented matrix norm for backward. Original Triton supports
fixed scaling only. Fixed-scaling correctness is restricted to the T18 norm
bound; larger norms are separately reported as stress cases. CUDA returns NaNs
for a matrix with nonfinite inputs without affecting neighboring matrices.
The standalone autograd wrapper supports first-order reverse mode.

Build and validate
------------------

Run from this directory with an installed CUDA PyTorch and CUDA toolkit. The
checkpoint was validated on PyTorch 2.14.1+cu130, Python 3.12, CUDA toolkit 13.1,
Triton 3.8.0, and A100 SM80. The MMA source requires an SM80-or-newer build target.
The repository uses ``pip install -e . -v --no-build-isolation`` for builds::

    export CUDA_HOME=/apps/external/cuda-toolkit/13.1.0
    export PATH="$CUDA_HOME/bin:$PATH"
    export TORCH_CUDA_ARCH_LIST='8.0+PTX'
    export MAX_JOBS=2
    python -m pip install -e . -v --no-build-isolation
    python test_matrix_exp.py -v
    python validate.py --backends scalar --output results/scalar_correctness.csv
    python validate.py --backends mma --dtype bf16 fp16 --output results/mma_correctness.csv

The extension uses optimized compilation with line information, preserved PTX
and cubins, and verbose assembler resource reports. Broad fast-math is disabled.
Dependencies are PyTorch, setuptools, ninja, NumPy, Triton, and expecttest.

The API is experiment-local::

    from matrix_exp_bench import expm, frechet

    y = expm(a, backend="scalar", scaling="adaptive")
    y_mma = expm(a, backend="mma", scaling="adaptive")
    derivative = frechet(a.mT, upstream, backend="mma", scaling="adaptive")

Inputs have trailing square dimensions N in {4, 8, 16, 32}; noncontiguous inputs
and arbitrary leading batch dimensions are accepted. The future ATen predicate
will cover three-dimensional [B,N,N] inputs with every N<=32, including padded
non-power-of-two dimensions. That integration is not part of this checkpoint.

Benchmark
---------

The timing command writes resumable CSV rows with three alternating trials,
median, mean, p99, variance, sample counts, and separate first-call latency::

    python benchmark.py --n 16 32 --dtype bf16 fp16 --batch 1 1024 65536
    python benchmark.py --backends scalar mma torch --scaling adaptive --output results/adaptive.csv

``warmup`` and ``rep`` are millisecond budgets, not iteration counts. Raw
measurements use preallocated buffers and CUDA graphs; wrappers include
conversions, allocations, and autograd in the combined operation. Compilation
and autotuning occur before steady-state timing. First-call measurements may
hit compiler caches and are not clean-cache build timings. Stock PyTorch uses
adaptive scaling, so its rows retain that label even in a fixed-scaling sweep.

The historical report and complete assembly/profiler archives remain in the
local experiment directory ``agent_space/matrix_exp_compare/tuning/mma``.
Generated results and build products are deliberately ignored here.
