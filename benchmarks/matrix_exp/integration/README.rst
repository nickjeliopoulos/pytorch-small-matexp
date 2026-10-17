ATen integration draft
======================

Installed-build validation
--------------------------

The public-API tests and timing scripts in this directory are part of the source
checkout. They do not need the standalone probe. After configuring the source
build environment, build from the repository root with::

    python -m pip install -e . -v --no-build-isolation

Run ``spin`` lint commands before rebuilding that checkout: their setup
regenerates ``torch/version.py`` without the CUDA build metadata. Running them
during or after the build can make an editable CUDA installation report
``torch.version.cuda is None``. The build restores the configured metadata;
verify it with the import check below before testing or building the probe.

Activate that installation's environment. Set ``repo`` to its source checkout
and run from outside the checkout so that Python cannot accidentally import a
different local ``torch`` directory::

    repo=/path/to/pytorch-source
    results="$repo/agent_space/matrix_exp_compare/source_build/validation"
    export PYTHONPATH="$repo/benchmarks/matrix_exp"
    export PYTORCH_TEST_WITH_SLOW=1
    export MATRIX_EXP_TRACE_DIR="$results/traces"
    cd /tmp
    python -c 'import torch; print(torch.__version__, torch.version.git_version, torch.__file__, torch._C.__file__, torch.version.cuda); print(torch.cuda.get_device_name())'
    python "$repo/benchmarks/matrix_exp/integration/test_native.py" -v
    python "$repo/test/test_linalg.py" -v -k matrix_exp
    python "$repo/test/test_ops_gradients.py" -v -k matrix_exp
    python "$repo/benchmarks/matrix_exp/test_validate.py" -v
    python "$repo/benchmarks/matrix_exp/integration/test_timing_metadata.py" -v
    python "$repo/benchmarks/matrix_exp/integration/validate_pytorch.py" --native --output "$results/errors"

``test_native.py`` uses the installed public API and ordinary autograd without
overrides. The core linalg suite also checks low-precision ``torch.func.grad``
and ``torch.func.jvp(torch.func.grad(...))`` against analytic diagonal cases.
Profiler tests require CUPTI and verify both forward and backward routing.
Numerical validation recomputes every case and overwrites its CSV; failures
produce a failing exit status. There is no correctness-resume option.
The same-build rank-4 FP32 path is a numerical control, not a timing backend.

Record ``git rev-parse HEAD``, ``git diff --binary``, untracked source files,
the build log, environment, and hashes of the tested source and loaded shared
libraries with each run. An editable build's version string identifies its
base commit; it does not identify subsequent uncommitted source changes.
Verify those changes were copied to the build checkout and recompiled.
Building PyTorch does not rebuild the separate probe described below.

Installed-build timing comparison
---------------------------------

For the installed custom PyTorch build, activate its environment and run::

    bash benchmarks/matrix_exp/integration/run_timings.sh /path/to/results /path/to/stock-venv/bin/python

The two backends are ``torch_small_n`` (our integrated CUDA kernel) and
``torch_linalg`` (the stock PyTorch 2.14.1 release). The runner executes three
trials in alternating order and generates forward/backward figures for FP32,
BF16, and FP16. Completed cases are resumable; use a fresh output directory
to collect new measurements. Both backends use identical saved input tensors.
The runner and summarizer require matching environment metadata across all
trials of each backend, including both CPU/autograd and CUDA library hashes.
Existing timing CSVs without their metadata are rejected.

``wall_summary.csv`` contains ``n``, ``dtype``, ``batch``, ``operation``,
``torch_small_n_median_ms``, ``torch_linalg_median_ms``, and ``speedup``.
Each reported time is the median of three trial medians. Speedup is
``torch_linalg_median_ms / torch_small_n_median_ms``. Detailed per-trial statistics
and all 25 samples remain in ``wall_timings.csv``.

``benchmark_native.py --backend torch_small_n|torch_linalg --trial 0 --output /path/to/results/timings``
runs one backend/trial; run ``torch_linalg`` first to generate the shared inputs.
The Python executable selects the installed package, so run ``torch_linalg``
with the stock environment and ``torch_small_n`` with the custom environment.
Set ``PYTHONPATH`` to ``benchmarks/matrix_exp`` for direct script invocation.
``summarize_native.py --output /path/to/results`` regenerates the timing report,
CSV summaries, and figures after the complete sweep. Numerical validation is
separate and is not required by the timing summarizer.

The timings include allocations, conversions, dispatch, and synchronization.
Backward times only ``autograd.grad`` with its forward graph prepared beforehand.
For BF16/FP16, ``torch_linalg`` computes in FP32 and casts back, including autograd
through those conversions. The standalone extension workflow below is separate
from this installed-public-API comparison.

Kernel integration
------------------

``aten/src/ATen/native/SmallMatrixExp.h`` defines the predicate: strided NVIDIA
CUDA inputs shaped [B,N,N], 2<=N<=32, with FP32, FP16, or BF16 dtype. N=0 and N=1
retain the existing trivial implementations. ``LinearAlgebra.cpp`` calls an
internal DispatchStub after its normal validation. No public operator schema,
dispatcher key, or Python registration changes are required.

``aten/src/ATen/native/cuda/MatrixExpKernel.cu`` registers the CUDA implementation.
The adjacent ``MatrixExpSmall.cuh`` contains the tiny and cooperative scalar
implementations. Both use FP32 scalar FMA products throughout the polynomial
and repeated squaring.
All calls use per-matrix adaptive scaling. Non-power-of-two dimensions are
zero-padded to 4/8/16/32 and cropped afterward; this initial wrapper incurs
conversion, allocation, and copy costs. Shape, device, and dtype are preserved.

FP32, FP16, and BF16 inputs use FP32 intermediates, with conversion to the input
dtype only at the output store. N<=8 uses the tiny kernel; larger supported
dimensions use the cooperative scalar kernel. The BF16 MMA implementation was
removed from ATen after large-angle rotation regressions exposed severe forward
and backward errors. Historical MMA experiments remain in the standalone
benchmark checkpoint. ROCm and inputs outside the predicate retain stock ATen.

``FunctionsManual.cpp`` uses the structured Frechet recurrence for ordinary
first derivatives. Higher derivatives retain the differentiable augmented
matrix construction, promoting low-precision augmented inputs to FP32 before
the exponential. Tensor subclasses, including ``torch.func`` wrappers, use
this differentiable operator-based path with the same precision policy.
Forward tangents on either argument also prevent the raw structured shortcut.

Isolated validation
-------------------

This probe compiles the exact new CUDA translation unit and extracts the edited
manual derivative and its existing helper from ``FunctionsManual.cpp``. It also
extracts the public entry-point body from ``LinearAlgebra.cpp``, replacing only
its file-local stock fallback with the installed wheel's native entry point.
It exercises the DispatchStub against an installed PyTorch wheel. It does not
compile all of ``LinearAlgebra.cpp``, generated autograd registrations, or
libtorch, and therefore is not a replacement for a full PyTorch build and CI.
The probe's CPU dispatch ABI flags target the x86-64 release wheel used here.
Build and test it separately; a successful full PyTorch rebuild does not
validate this generated translation unit. Its extracted derivative imports
the autograd forward-tangent helper explicitly.

From this directory, with the checkpoint package installed and the same CUDA
environment active::

    export MAX_JOBS=2
    export TORCH_CUDA_ARCH_LIST='7.5;8.0+PTX'
    python -m pip install -e . -v --no-build-isolation
    python test_integration.py -v

Tests compare all dimensions 2 through 32 against FP64, including empty batches,
tails, noncontiguous inputs, adaptive boundaries, nonfinite values, graph capture,
and directional finite differences. A scoped CUDA override exercises public
matrix_exp aliases and the existing wheel's generated autograd. A separate
autograd wrapper exercises the compiled edited manual derivative, including
second derivatives. Overrides live only in the test process and are removed
when each test context exits.

Profiler-based dispatch tests check N=0/1/2, 8/9, 15/16, and 32/33. Eligible
inputs must launch exactly one kernel in the expected tiny or scalar
namespace; ineligible sizes must launch none of these kernels. CUDA kernel
tracing requires CUPTI. The corresponding core test is
``test_linalg_matrix_exp_small_cuda_dispatch`` in ``test/test_linalg.py`` checks
both forward and backward. ``test_linalg_matrix_exp_large_rotation`` checks
analytic forward and backward values and orthogonality at angles 25, 100, and
1000 for N=8/16/32, all three dtypes, and batches 1/129.

Historically, on gilbreth-i004 (A100 SM80), all 19 probe tests passed after the CuTe migration.
Both SM75 and SM80 targets compiled; SM75 was not tested on hardware.
The new core dispatch test body also passed against the extracted entry point
for all three dtypes. ``torch.compile`` was tested
with the eager backend, alongside export and vmap, rather than an Inductor
performance run. The available lint checks passed with Clang-tidy excluded.

Build products, extracted derivative source, and assembly are saved in ignored
``artifacts``. Full build and test logs remain under the existing experiment's
``agent_space/matrix_exp_compare/results/aten_*`` files. The added core test
``test_linalg_matrix_exp_small_cuda`` is intended for a full rebuilt PyTorch.
Clang-tidy also requires that full build's compilation database; other checks
can be run through ``spin quicklint -- --skip CLANGTIDY <changed paths>``.

Comparing implementation revisions
---------------------------------

Build each revision in a separate source tree and retain its extension binary.
Load each binary in its own process, since both register the same probe ops::

    python compare.py validate --library /path/to/baseline.so --label baseline --output /path/to/results
    python compare.py validate --library /path/to/candidate.so --label candidate --output /path/to/results --reference-outputs /path/to/results/baseline_outputs.pt
    python compare.py benchmark --library /path/to/baseline.so --label baseline --output /path/to/results --trial 0
    python compare.py benchmark --library /path/to/candidate.so --label candidate --output /path/to/results --trial 0

Repeat timing trials with alternating revision order. Defaults cover N=4/8/16/32,
FP32/BF16/FP16, and batches 1/32/1024/65536, with adaptive scaling. Validation
records FP64-reference errors per case and exact output comparisons with the
baseline. Forward, backward-only, and forward-plus-autograd-backward are timed
separately. CUDA graph timings include the wrappers' GPU conversions and copies;
allocations are captured. Synchronized wall timings also include allocation,
Python dispatch, and autograd overhead. These are not isolated raw-kernel timings.
Compilation is excluded. CSVs contain median, mean, p99, and variance; metadata
records the loaded binary's SHA256 and environment.

The CuTe comparison against commit ``a8d94efe2d2`` passed all 264 reference cases,
with bitwise-identical outputs between revisions. Three alternating-order
trials gave a median CUDA-graph latency change of 0%; individual cases ranged
from 2.36% faster to 1.69% slower. Wall-clock changes were more variable,
including for unchanged scalar kernels. The detailed report, CSVs, figures,
source snapshots, and assembly are in the ignored experiment directory
``agent_space/matrix_exp_compare/cute_migration/``.

Forward and backward wall time against default PyTorch
-----------------------------------------------------

``compare_pytorch.py`` compares the compiled current ATen entry point against
the unmodified release wheel, without installing a public operator override::

    python compare_pytorch.py --library /path/to/probe.so --output /path/to/results

It measures wall time with CUDA synchronization, including allocations and
copies. The ``operation`` column contains ``fwd`` or ``bwd``. Forward rows time
only the matrix exponential on inputs without gradients. Backward rows time
``torch.autograd.grad(..., retain_graph=True)`` with a random upstream gradient;
the forward graph is prepared outside timing and reused across samples. These
are backward-only timings, not forward-plus-backward or ``.grad`` accumulation.
Our probe reaches the compiled manual derivative through a Python
``autograd.Function``; its overhead is included in backward wall time.
Use ``--operation fwd`` or ``--operation bwd`` to restrict the sweep.
Defaults are three alternating-order trials of 25 calls for each
N=4/8/16/32 and B=1/32/256/1024/2048/4096/8192/16384/32768/65536/120064.
FP32 uses ``torch.linalg.matrix_exp(a)`` directly. The separately labeled BF16
and FP16 baselines use ``torch.linalg.matrix_exp(a.float()).to(a.dtype)``, with
both casts included in the timing. The integrated kernel also uses FP32
intermediates. All cases use identical prescribed-1-norm-3 inputs.
Results are saved per operation/trial/backend and can be resumed with the same binary,
PyTorch version, GPU model, and sample count.
Older forward-only CSVs are migrated by adding ``operation=fwd`` to their rows.

The historical A100 measurements below predate removal of BF16 MMA from ATen.
The absolute-time and speedup figures for each dtype, and
complete CSVs are in ``agent_space/matrix_exp_compare/cute_migration/pytorch_wall/``.
Across the FP32 sweep, our synchronized forward wall time was 9.05x to 75.26x
faster. These measurements include dispatch and launch savings; they are not
raw-kernel speedup claims.

Numerical comparison against default PyTorch
-------------------------------------------

``test_autograd_against_pytorch`` calls ``.backward()`` on both our integration
wrapper and stock PyTorch. Its 12 parameterized tests cover FP32/BF16/FP16,
N=4/8/16/32, batches 0/1/35/129, noncontiguous inputs, and eight input families
or norm cases. The stock low-precision policy matches the timing baseline.

``validate_pytorch.py`` records numerical errors for the full timing grid plus
random, skew-symmetric, negative semidefinite, diagonal, zero, identity, and
prescribed-norm cases, including adaptive-scaling boundaries. Both backwards
run through ``torch.autograd.grad`` with the same random upstream gradient::

    python test_integration.py -v -k autograd_against_pytorch
    python validate_pytorch.py --library /path/to/probe.so --output /path/to/errors

The CSV reports maximum absolute error, global magnitude-normalized error,
worst and p95 per-matrix normalized errors, and worst matrix index. Per-matrix
error divides the maximum absolute difference by one plus the maximum absolute
reference value in that matrix. Every matrix is compared against stock PyTorch.
FP64 checks additionally cover all family-sweep matrices and the first 32
matrices of each timing-grid case; their sample counts are explicit in the CSV.

Forward acceptance tolerances are 0.001 (FP32) and 0.05 (BF16/FP16); backward
tolerances are 0.005 and 0.1, respectively, retaining the integration test's
experimental precision policy. The command exits unsuccessfully if any
comparison fails and saves the failing inputs and outputs. Detailed results
are in ``agent_space/matrix_exp_compare/cute_migration/pytorch_errors/``.
