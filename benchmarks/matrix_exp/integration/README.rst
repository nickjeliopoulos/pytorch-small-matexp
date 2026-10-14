ATen integration draft
======================

``aten/src/ATen/native/SmallMatrixExp.h`` defines the predicate: strided NVIDIA
CUDA inputs shaped [B,N,N], 2<=N<=32, with FP32, FP16, or BF16 dtype. N=0 and N=1
retain the existing trivial implementations. ``LinearAlgebra.cpp`` calls an
internal DispatchStub after its normal validation. No public operator schema,
dispatcher key, or Python registration changes are required.

``aten/src/ATen/native/cuda/MatrixExpKernel.cu`` registers the CUDA implementation.
The adjacent ``MatrixExpSmall.cuh`` contains the row and cooperative scalar
implementations; ``MatrixExpMma.cuh`` contains the MMA implementation.
The MMA implementation uses the pinned CUTLASS submodule's CuTe SM80 BF16
MMA and SM75 matrix-load intrinsics. Fragment layouts and launch geometry
remain local to the matrix exponential kernel.
All calls use per-matrix adaptive scaling. Non-power-of-two dimensions are
zero-padded to 4/8/16/32 and cropped afterward; this initial wrapper incurs
conversion, allocation, and copy costs. Shape, device, and dtype are preserved.

FP32 uses scalar arithmetic. FP16 and BF16 use FP32 scalar intermediates below
N=16 and BF16 MMA operands with FP32 accumulation from N=16 on SM80 or newer.
BF16 intermediates with FP16 output are explicitly permitted by the experimental
precision policy. Older devices, and PTX compiled for an older target, use
scalar arithmetic. ROCm and inputs outside the predicate retain stock ATen.
Low-precision MMA accuracy remains experimental; dtype preservation does not
establish upstream numerical acceptance.

``FunctionsManual.cpp`` uses the structured Frechet recurrence for ordinary
first derivatives. Higher derivatives retain the differentiable augmented
matrix construction, promoting low-precision augmented inputs to FP32 before
the exponential. Tensor subclasses retain the original path. These guards
avoid replacing higher derivatives with a non-differentiable custom launch.

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

From this directory, with the checkpoint package installed and the same CUDA
environment active::

    export MAX_JOBS=2
    export TORCH_CUDA_ARCH_LIST='7.5;8.0+PTX'
    git -C ../../.. submodule update --init third_party/cutlass
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
inputs must launch exactly one kernel in the expected tiny, scalar, or MMA
namespace; ineligible sizes must launch none of these kernels. CUDA kernel
tracing requires CUPTI. The corresponding core test is
``test_linalg_matrix_exp_small_cuda_dispatch`` in ``test/test_linalg.py``.

On gilbreth-i004 (A100 SM80), all 19 probe tests passed after the CuTe migration.
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
both casts included in the timing. Their internal precision therefore differs
from our BF16 MMA path. All cases use identical prescribed-1-norm-3 inputs.
Results are saved per operation/trial/backend and can be resumed with the same binary,
PyTorch version, GPU model, and sample count.
Older forward-only CSVs are migrated by adding ``operation=fwd`` to their rows.

The A100 measurements, absolute-time and speedup figures for each dtype, and
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
