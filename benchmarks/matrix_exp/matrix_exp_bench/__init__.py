"""Standalone matrix-exponential research kernels; no ATen overrides."""

import importlib.util
from functools import lru_cache

import torch


DTYPES = (torch.float32, torch.float16, torch.bfloat16)
SIZES = (4, 8, 16, 32)
THETA18 = 3.010066362817634


@lru_cache(None)
def _operators():
    spec = importlib.util.find_spec("matrix_exp_bench._C")
    if spec is None:
        raise RuntimeError(
            "Build the extension first; see benchmarks/matrix_exp/README.rst"
        )
    torch.ops.load_library(spec.origin)
    return torch.ops.matrix_exp_bench


def _validate(a, backend, scaling, S):
    if backend not in ("scalar", "mma", "triton_original", "torch"):
        raise ValueError(f"Unknown backend: {backend}")
    if (
        scaling not in ("fixed", "adaptive")
        or not isinstance(S, int)
        or not 0 <= S <= 512
    ):
        raise ValueError("Expected fixed/adaptive scaling and an integer S in [0, 512]")
    if not a.is_cuda or a.dtype not in DTYPES:
        raise ValueError("Expected a CUDA FP32, FP16 or BF16 tensor")
    if a.ndim < 2 or a.shape[-2] != a.shape[-1] or a.shape[-1] not in SIZES:
        raise ValueError("Expected (..., N, N), N in {4, 8, 16, 32}")
    if backend == "triton_original" and scaling != "fixed":
        raise ValueError("Original hyperconnections Triton supports fixed scaling only")


def _threads(n, batch, pair, backend):
    if backend == "scalar":
        if n == 16:
            return 256 if batch <= 32 else 128 if batch <= 128 else 64
        return 256 if batch <= 32 else 128
    if n == 16:
        return 64 if pair and batch <= 128 else 32
    if pair:
        return 128 if batch <= 128 else 64
    return 128 if batch <= 32 else 64 if batch <= 2048 else 32


def raw(x, g, out, backend="scalar", scaling="fixed", S=2):
    """Launch with contiguous FP32 inputs and a preallocated output."""
    if not x.numel():
        return out
    n, batch = x.shape[-1], x.shape[0]
    if backend in ("scalar", "mma"):
        ops = _operators()
        if n < 16:
            ops.rows_out(x, g, out, scaling == "adaptive", S)
        else:
            if backend == "mma" and torch.cuda.get_device_capability(x.device)[0] < 8:
                raise ValueError("BF16 MMA requires an SM80 or newer CUDA device")
            op = ops.mma_out if backend == "mma" else ops.scalar_out
            op(
                x,
                g,
                out,
                scaling == "adaptive",
                S,
                _threads(n, batch, g is not None, backend),
            )
    elif backend == "triton_original":
        if scaling != "fixed":
            raise ValueError("Original Triton supports fixed scaling only")
        from .reference import expm_triton as kernels

        kw = dict(
            N=n,
            NP=n,
            MAX_S=S,
            OUT_DTYPE=kernels._TORCH_TO_TL[out.dtype],
            LOW_PREC=out.dtype in (torch.float16, torch.bfloat16),
        )
        with torch.cuda.device(x.device):
            if g is None:
                kernels._expm_t18_fwd[(batch,)](
                    x, out, 2.0 ** (-S), *x.stride(), *out.stride(), **kw
                )
            else:
                kernels._expm_t18_structured_fwd[(batch,)](
                    x,
                    g,
                    out,
                    2.0 ** (-S),
                    *x.stride(),
                    *g.stride(),
                    *out.stride(),
                    **kw,
                )
    else:
        raise ValueError("Raw timing requires a native CUDA or Triton backend")
    return out


def _forward(a, backend, scaling, S):
    _validate(a, backend, scaling, S)
    if backend == "torch":
        return torch.linalg.matrix_exp(a.float()).to(a.dtype)
    n = a.shape[-1]
    x = a.float().contiguous().reshape(-1, n, n)
    out = torch.empty_like(x, dtype=a.dtype)
    return raw(x, None, out, backend, scaling, S).reshape(a.shape)


def frechet(x, g, backend="scalar", scaling="fixed", S=2):
    """L_exp(X, G). Reverse-mode differentiation supplies X=A.T."""
    _validate(x, backend, scaling, S)
    if x.shape != g.shape or x.device != g.device or g.dtype not in DTYPES:
        raise ValueError("G must match X shape/device and have a supported dtype")
    n = x.shape[-1]
    if backend == "torch":
        xf, gf = x.float(), g.float()
        z = torch.zeros_like(xf)
        block = torch.cat((torch.cat((xf, gf), -1), torch.cat((z, xf), -1)), -2)
        return torch.linalg.matrix_exp(block)[..., :n, n:].to(x.dtype)
    xf = x.float().contiguous().reshape(-1, n, n)
    gf = g.float().contiguous().reshape(-1, n, n)
    out = torch.empty_like(xf, dtype=x.dtype)
    return raw(xf, gf, out, backend, scaling, S).reshape(x.shape)


class _Expm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, backend, scaling, S):
        ctx.save_for_backward(a)
        ctx.backend, ctx.scaling, ctx.S = backend, scaling, S
        return _forward(a, backend, scaling, S)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        (a,) = ctx.saved_tensors
        return frechet(a.mT, grad, ctx.backend, ctx.scaling, ctx.S), None, None, None


def expm(a, backend="scalar", scaling="fixed", S=2):
    if backend == "torch":
        return _forward(a, backend, scaling, S)
    return _Expm.apply(a, backend, scaling, S)
