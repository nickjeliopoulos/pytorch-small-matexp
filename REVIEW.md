# Small CUDA matrix exponential: PR review guide

These checks apply to this PR's matrix-exponential changes. They do not add
requirements for unrelated PyTorch code. See
[`benchmarks/matrix_exp/integration/README.rst`](benchmarks/matrix_exp/integration/README.rst)
for runnable validation and timing commands.

## Scope and entry points

- `aten/src/ATen/native/SmallMatrixExp.h`: eligibility predicate and DispatchStub.
- `aten/src/ATen/native/LinearAlgebra.cpp`: public forward routing.
- `aten/src/ATen/native/cuda/MatrixExpKernel.cu`: conversions, padding, device and
  stream handling, launch selection, and CUDA registration.
- `aten/src/ATen/native/cuda/MatrixExpSmall.cuh`: FP32 polynomial, adaptive
  scaling, squaring, and structured Fréchet derivative kernels.
- `torch/csrc/autograd/FunctionsManual.cpp`: first derivatives and the
  differentiable augmented-matrix fallback.
- `test/test_linalg.py`: public-API correctness, dispatch, and AD regressions.
- `benchmarks/matrix_exp/integration/`: standalone probe and installed-build
  validation, numerical comparison, and wall-time scripts.

## Contracts to preserve

1. Eligible inputs are strided NVIDIA CUDA tensors shaped `[B, N, N]`, with
   `2 <= N <= 32` and FP32, FP16, or BF16 dtype. Check fallback behavior for CPU,
   ROCm, other dtypes, other batch ranks, and unsupported dimensions. Preserve
   public validation and the existing `N=0` and `N=1` cases.
2. Outputs and gradients preserve shape, dtype, and device. All polynomial and
   squaring products use FP32 intermediates. The removed BF16 operand-rounding
   MMA path must not return without evidence covering large-angle rotations.
3. Norm-based scaling is per matrix, including mixed norms in one batch. The
   backward norm covers the represented augmented matrix. Check scaling
   boundaries, nonfinite termination, empty batches, and partial blocks.
4. Raw pointers must see logical tensor values. In particular, conversion and
   contiguity alone do not resolve a float32 tensor's lazy negative bit.
5. The raw structured derivative is only for ordinary first derivatives:
   reverse-mode recording, forward tangents on either argument, and tensor
   subclasses must retain a differentiable operator-based path. Low-precision
   promotion on that path must also work with `torch.func` wrappers. Eligibility
   checks must preserve symbolic batch dimensions instead of specializing them.
6. Device guards and launches must use the correct CUDA device and current
   stream; graph capture must not introduce host synchronization.

## Evidence required for review

- Test the rebuilt public `torch.linalg.matrix_exp` and generated autograd,
  without Python overrides or the standalone probe loaded into that process.
  Trace forward and backward: tiny kernels for `N<=8`, scalar kernels above,
  and no small kernel where the eligibility predicate excludes the input.
- Check FP32/BF16/FP16 against FP64 forward and augmented-matrix derivative
  references, with random upstream gradients. Include per-matrix errors,
  finite differences, noncontiguous and lazy-negative inputs, analytic diagonal
  cases, and rotations at angles 25, 100, and 1000 for `N=16,32`.
- Exercise ordinary backward, forward AD, higher-order reverse AD,
  forward-over-reverse AD, and `torch.func.grad` / `torch.func.jvp(grad)`.
  At `A=0`, the gradient of `trace(exp(A))` and its derivative in direction `I`
  both equal `I`. Double-only gradchecks cannot cover this CUDA fast path.
- Regenerate correctness results from scratch. Keep failures visible, return
  a failing status, and invalidate completed summaries if a run is interrupted.
  Acceptance limits remain 0.001/0.005 for FP32 forward/backward and 0.05/0.1
  for low precision. Stress cases and tolerances must be labeled explicitly.
- Rebuild the standalone probe separately when its source extraction changes.
  Its generated derivative needs the correct namespace and headers for
  autograd helpers. Passing the full build does not establish probe validity;
  passing the probe does not establish public-API integration.
- Record source hashes, the source/build checkout revisions and local patches,
  Python/package paths, shared-library hashes, CUDA configuration, commands,
  exit statuses, and logs. A stale base revision in `torch.version.git_version`
  cannot by itself prove which patched source was compiled.
- Keep reusable tests and runners in the tracked repository. Put binaries,
  traces, CSVs, plots, and machine-specific logs in ignored `agent_space/`.

## Performance comparisons

Compare `torch_small_n` with default `torch_linalg` in separate installations.
Label FP16/BF16 default-PyTorch measurements as FP32 computation plus input and
output casts, including autograd through those casts. Use identical saved
inputs and upstream gradients, three alternating trials, and 25 synchronized
wall-time samples. Report `operation=fwd` or `bwd`; backward excludes preparation
of its forward graph. Include wrapper costs and record build/version differences.
Do not present historical MMA measurements as results for the current FP32
implementation or use timing success as evidence of numerical correctness.
