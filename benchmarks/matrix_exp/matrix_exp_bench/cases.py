import torch


DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def make_input(b, n, dtype, kind="random", norm=5.0, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    # Match the hyperconnections factories, including low-precision generation.
    x = torch.randn(b, n, n, generator=gen, dtype=dtype, device=device)
    if kind == "random":
        return x * 0.3
    if kind == "skew":
        x = x * 0.5
        return 0.5 * (x - x.mT)
    if kind == "neg_psd":
        x = x * 0.4
        return -(x @ x.mT)
    if kind == "diagonal":
        d = torch.randn(b, n, generator=gen, dtype=dtype, device=device) * 0.5
        return torch.diag_embed(d)
    if kind == "large":
        x = x.float()
        scale = torch.linalg.matrix_norm(x, ord=1).clamp_min(1e-12)
        return (x * (norm / scale).view(-1, 1, 1)).to(dtype)
    if kind == "zero":
        return torch.zeros_like(x)
    if kind == "identity":
        return torch.eye(n, device=device, dtype=dtype).expand(b, n, n).clone()
    raise ValueError(kind)


def reference(a, g=None):
    a = a.double()
    if g is None:
        return torch.linalg.matrix_exp(a)
    n = a.shape[-1]
    g = g.double()
    z = torch.zeros_like(a)
    m = torch.cat((torch.cat((a, g), -1), torch.cat((z, a), -1)), -2)
    return torch.linalg.matrix_exp(m)[..., :n, n:]


def metrics(got, expected):
    ref = expected.double()
    diff = (got.double() - ref).abs()
    per_err = diff.flatten(-2).amax(-1)
    per_mag = ref.abs().flatten(-2).amax(-1)
    rel = per_err / (1 + per_mag)
    valid = torch.isfinite(got).flatten(-2).all(-1)
    rel = torch.where(valid, rel, float("inf"))
    return dict(
        max_err=per_err.max().item(),
        global_rel_err=(per_err.max() / (1 + per_mag.max())).item(),
        worst_matrix_rel_err=rel.max().item(),
        worst_matrix_index=rel.argmax().item(),
        finite=bool(valid.all().item()),
    )
