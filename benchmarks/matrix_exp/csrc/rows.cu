#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace {
constexpr int threads = 128;
constexpr double theta18 = 3.010066362817634;

template <int N, bool Pair>
struct Tile {
  float d[N];
  float u[Pair ? N : 1];
};

template <int N, bool Pair>
__device__ __forceinline__ Tile<N, Pair> product(
    const Tile<N, Pair>& a, const Tile<N, Pair>& b, unsigned mask) {
  Tile<N, Pair> c;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    float d = 0.f;
    float u = 0.f;
#pragma unroll
    for (int k = 0; k < N; ++k) {
      const float bd = __shfl_sync(mask, b.d[j], k, N);
      d = fmaf(a.d[k], bd, d);
      if constexpr (Pair) {
        const float b_upper = __shfl_sync(mask, b.u[j], k, N);
        // Separate products match the structured Triton algebra.
        u = fmaf(a.d[k], b_upper, u);
      }
    }
    c.d[j] = d;
    if constexpr (Pair) {
      float v = 0.f;
#pragma unroll
      for (int k = 0; k < N; ++k) {
        const float bd = __shfl_sync(mask, b.d[j], k, N);
        v = fmaf(a.u[k], bd, v);
      }
      c.u[j] = u + v;
    }
  }
  return c;
}

template <int N, bool Pair>
__device__ __forceinline__ Tile<N, Pair> add(
    const Tile<N, Pair>& a, const Tile<N, Pair>& b) {
  Tile<N, Pair> c;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    c.d[j] = a.d[j] + b.d[j];
    if constexpr (Pair) c.u[j] = a.u[j] + b.u[j];
  }
  return c;
}

template <int N, bool Pair>
__device__ __forceinline__ Tile<N, Pair> combine(
    const Tile<N, Pair>& a, const Tile<N, Pair>& a2,
    const Tile<N, Pair>& a3, const Tile<N, Pair>& a6,
    float c0, float c1, float c2, float c3, float c6, int row) {
  Tile<N, Pair> b;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    b.d[j] = c0 * float(row == j) + c1*a.d[j] + c2*a2.d[j]
             + c3*a3.d[j] + c6*a6.d[j];
    if constexpr (Pair) {
      b.u[j] = c1*a.u[j] + c2*a2.u[j] + c3*a3.u[j] + c6*a6.u[j];
    }
  }
  return b;
}

template <int N, bool Pair>
__device__ __forceinline__ Tile<N, Pair> t18(
    const Tile<N, Pair>& a, unsigned mask, int row) {
  auto a2 = product(a, a, mask);
  auto a3 = product(a2, a, mask);
  auto a6 = product(a3, a3, mask);
  auto b1 = combine(a, a2, a3, a6, 0.f,
      -0.10036558103014462f, -0.008029246482411570f,
      -0.000892138498045730f, 0.f, row);
  auto b2 = combine(a, a2, a3, a6, 0.f,
      0.39784974949645076f, 1.36783778460411719f,
      0.49828962252538268f, -0.00063789819459247233f, row);
  auto b3 = combine(a, a2, a3, a6, -10.96763960529620626f,
      1.68015813878906197f, 0.05717798464788655f,
      -0.00698210122488052f, 3.349750170860705e-05f, row);
  auto b4 = combine(a, a2, a3, a6, -0.09043168323908106f,
      -0.06764045190713819f, 0.06759613017740597f,
      0.02955525704293155f, -1.391802575160607e-05f, row);
  auto b5 = combine(a, a2, a3, a6, 0.f, 0.f,
      -0.09233646193671186f, -0.01693649390020817f,
      -1.400867981820361e-05f, row);
  auto a9 = add(product(b1, b5, mask), b4);
  return add(b2, product(add(b3, a9), a9, mask));
}

__device__ __forceinline__ void store_value(void* out, int dtype, int64_t i, float v) {
  if (dtype == 0) static_cast<float*>(out)[i] = v;
  else if (dtype == 1) static_cast<c10::BFloat16*>(out)[i] = c10::BFloat16(v);
  else static_cast<c10::Half*>(out)[i] = c10::Half(v);
}

// Mode 2 specializes the original S=2 baseline; -1 is adaptive; -2 is fixed runtime S.
template <int N, bool Pair, int Mode>
__global__ void expm_kernel(const float* x, const float* g, void* out,
                          int dtype, int64_t batch, int requested_s) {
  const int lane = threadIdx.x & 31;
  const int row = lane % N;
  const int group_start = lane - row;
  const unsigned mask = (0xffffffffu >> (32 - N)) << group_start;
  const int64_t b = (int64_t(blockIdx.x)*threads + threadIdx.x)/N;
  if (b >= batch) return;
  const int64_t base = b*N*N + row*N;
  Tile<N, Pair> a;
  bool bad = false;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    a.d[j] = x[base+j];
    bad |= !isfinite(a.d[j]);
    if constexpr (Pair) {
      a.u[j] = g[base+j];
      bad |= !isfinite(a.u[j]);
    }
  }
  if (__any_sync(mask, bad)) {
#pragma unroll
    for (int j = 0; j < N; ++j) store_value(out, dtype, base+j, NAN);
    return;
  }
  int s;
  if constexpr (Mode == -1) {
    // FP64 norm accumulation avoids overflow on finite FP32 entries.
    double norm = 0.;
#pragma unroll
    for (int j = 0; j < N; ++j) {
      double col = fabs(double(a.d[j]));
      if constexpr (Pair) col += fabs(double(a.u[j]));
#pragma unroll
      for (int offset = N/2; offset; offset /= 2)
        col += __shfl_xor_sync(mask, col, offset, N);
      norm = fmax(norm, col);
    }
    s = norm > theta18 ? int(ceil(log2(norm/theta18))) : 0;
  } else if constexpr (Mode == 2) s = 2;
  else s = requested_s;
  const float scale = ldexpf(1.f, -s);
#pragma unroll
  for (int j = 0; j < N; ++j) {
    a.d[j] *= scale;
    if constexpr (Pair) a.u[j] *= scale;
  }
  auto result = t18(a, mask, row);
  if constexpr (Mode == 2) {
#pragma unroll
    for (int i = 0; i < 2; ++i) result = product(result, result, mask);
  } else {
    for (int i = 0; i < s; ++i) result = product(result, result, mask);
  }
#pragma unroll
  for (int j = 0; j < N; ++j)
    store_value(out, dtype, base+j, Pair ? result.u[j] : result.d[j]);
}

template <int N, bool Pair>
void launch(const at::Tensor& x, const float* g, const at::Tensor& out,
            bool adaptive, int s, int dtype) {
  const int64_t batch = x.size(0);
  const dim3 grid((batch + threads/N - 1)/(threads/N));
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (adaptive)
    expm_kernel<N, Pair, -1><<<grid, threads, 0, stream>>>(
        x.data_ptr<float>(), g, out.data_ptr(), dtype, batch, s);
  else if (s == 2)
    expm_kernel<N, Pair, 2><<<grid, threads, 0, stream>>>(
        x.data_ptr<float>(), g, out.data_ptr(), dtype, batch, s);
  else
    expm_kernel<N, Pair, -2><<<grid, threads, 0, stream>>>(
        x.data_ptr<float>(), g, out.data_ptr(), dtype, batch, s);
}
} // namespace

void matrix_exp_out(const at::Tensor& x, const std::optional<at::Tensor>& g,
                    const at::Tensor& out, bool adaptive, int64_t s) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.is_contiguous(),
              "raw input must be contiguous CUDA FP32");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == x.size(2), "expected [B,N,N]");
  TORCH_CHECK(out.device() == x.device() && out.sizes() == x.sizes()
              && out.is_contiguous(), "output shape/device/layout mismatch");
  TORCH_CHECK(s >= 0 && s <= 512, "fixed S must be in [0,512]");
  int dtype = out.scalar_type() == at::kFloat ? 0 :
              out.scalar_type() == at::kBFloat16 ? 1 :
              out.scalar_type() == at::kHalf ? 2 : -1;
  TORCH_CHECK(dtype >= 0, "output must be float32, bfloat16 or float16");
  if (g) TORCH_CHECK(g->sizes() == x.sizes() && g->device() == x.device()
      && g->scalar_type() == at::kFloat && g->is_contiguous(), "invalid raw G");
  const auto n = x.size(1);
  TORCH_CHECK(n == 4 || n == 8 || n == 16 || n == 32, "unsupported N");
#ifdef MATEXP_CORE_ONLY
  TORCH_CHECK(n == 4 || n == 8, "the core-only build supports N=4 and N=8; finish the full build for larger N");
#endif
  if (x.numel() == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
#define LAUNCH(N) if (g) launch<N, true>(x, g->data_ptr<float>(), out, adaptive, s, dtype); \
                  else launch<N, false>(x, nullptr, out, adaptive, s, dtype)
  switch (n) {
    case 4: { LAUNCH(4); break; }
    case 8: { LAUNCH(8); break; }
#ifndef MATEXP_CORE_ONLY
    case 16: { LAUNCH(16); break; }
    case 32: { LAUNCH(32); break; }
#endif
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
