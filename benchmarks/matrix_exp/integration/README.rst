ATen integration draft
======================

The standalone comparison is checkpointed in commit ``3769b2fd622``. The
subsequent ATen changes are separate, uncommitted work for review.

``aten/src/ATen/native/SmallMatrixExp.h`` defines the predicate: strided NVIDIA
CUDA inputs shaped [B,N,N], 2<=N<=32, with FP32, FP16, or BF16 dtype. N=0 and N=1
retain the existing trivial implementations. ``LinearAlgebra.cpp`` calls an
internal DispatchStub after its normal validation. No public operator schema,
dispatcher key, or Python registration changes are required.

``aten/src/ATen/native/cuda/MatrixExpKernel.cu`` registers the CUDA implementation.
The adjacent Rows, Scalar, and Mma headers contain the checkpoint algorithms.
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
manual derivative and its existing helper from ``FunctionsManual.cpp``. It
exercises the DispatchStub against an installed PyTorch wheel. It does not
compile all of ``LinearAlgebra.cpp``, generated autograd registrations, or
libtorch, and therefore is not a replacement for a full PyTorch build and CI.
The probe's CPU dispatch ABI flags target the x86-64 release wheel used here.

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

On gilbreth-k014 (A100 SM80), all 16 probe tests passed. Both SM75 and SM80
targets compiled; SM75 was not tested on hardware. ``torch.compile`` was tested
with the eager backend, alongside export and vmap, rather than an Inductor
performance run. The available lint checks passed with Clang-tidy excluded.

Build products, extracted derivative source, and assembly are saved in ignored
``artifacts``. Full build and test logs remain under the existing experiment's
``agent_space/matrix_exp_compare/results/aten_*`` files. The added core test
``test_linalg_matrix_exp_small_cuda`` is intended for a full rebuilt PyTorch.
Clang-tidy also requires that full build's compilation database; other checks
can be run through ``spin quicklint -- --skip CLANGTIDY <changed paths>``.

No GitHub fork is required for local development or commits. Publishing an
upstream PR normally requires a remote fork and a human-reviewed contribution.
Nothing has been pushed or submitted.
