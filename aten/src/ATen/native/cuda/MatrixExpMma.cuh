// Adapted from the validated benchmarks/matrix_exp checkpoint.
#pragma once

#include <ATen/native/cuda/MatrixExpSmall.cuh>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/bit_cast.h>
#include <cuda_bf16.h>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cmath>

namespace at::native::small_matrix_exp::mma {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162_raw packed = __floats2bfloat162_rn(lo, hi);
  return c10::bit_cast<uint32_t>(packed);
}

__device__ __forceinline__ void mma16816(
    float* c, const uint32_t* a, const uint32_t* b) {
  cute::SM80_16x8x16_F32BF16BF16F32_TN::fma(
      c[0], c[1], c[2], c[3],
      a[0], a[1], a[2], a[3],
      b[0], b[1],
      c[0], c[1], c[2], c[3]);
}

template <int N, int Warps, bool Pair>
struct MmaOps {
  static constexpr int WarpM = Warps < N / 16 ? Warps : N / 16;
  static constexpr int WarpN = Warps / WarpM;
  static constexpr int M = (N / 16) / WarpM;
  static constexpr int C = (N / 8) / WarpN;
  static constexpr int Elements = M * C * 4;
  static constexpr int Stride = N + 8;
  static constexpr int Plane = N * Stride;
  static constexpr bool DirectA = WarpN == 1;
  static constexpr int SharedElements = Plane * (Pair ? 2 : 1) * (DirectA ? 1 : 2);
  using Ops = MmaOps<N, Warps, Pair>;
  struct Tile {
    float d[Elements];
    float u[Pair ? Elements : 1];
  };
  uint16_t* shared;

  __device__ __forceinline__ static int index(int e) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int tile = e / 4;
    const int row =
        ((tile / C) * WarpM + warp % WarpM) * 16 + lane / 4 + ((e % 4) / 2) * 8;
    const int col = ((tile % C) * WarpN + warp / WarpM) * 8 + (lane % 4) * 2 + e % 2;
    return row * N + col;
  }

  __device__ __forceinline__ static void sync() {
    if constexpr (Warps == 1)
      __syncwarp();
    else
      __syncthreads();
  }

  __device__ __forceinline__ void stage(const float* x, int plane) const {
    #pragma unroll
    for (int e = 0; e < Elements; e += 2) {
      const int i = index(e);
      *reinterpret_cast<uint32_t*>(
          shared + plane * Plane + (i / N) * Stride + i % N) =
          pack_bf16(x[e], x[e + 1]);
    }
  }

  __device__ __forceinline__ void load_b(
      uint32_t* b, int k, int col, int plane) const {
    const int lane = threadIdx.x % 32;
    const auto* src = reinterpret_cast<const cute::uint128_t*>(
        shared + plane * Plane + (k * 16 + lane % 16) * Stride + col * 8);
    cute::SM75_U16x4_LDSM_T::copy(*src, b[0], b[1]);
  }

  __device__ __forceinline__ void load_b_pair(
      uint32_t* b, int k, int col, int plane) const {
    const int lane = threadIdx.x % 32;
    const auto* src = reinterpret_cast<const cute::uint128_t*>(
        shared + plane * Plane + (k * 16 + lane % 16) * Stride +
        (col + (lane / 16) * WarpN) * 8);
    cute::SM75_U16x8_LDSM_T::copy(*src, b[0], b[1], b[2], b[3]);
  }

  __device__ __forceinline__ void load_a(
      uint32_t* a, const float* x, int row, int k, int plane) const {
    if constexpr (DirectA) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int e = (row * C + k * 2 + j / 2) * 4 + (j % 2) * 2;
        a[j] = pack_bf16(x[e], x[e + 1]);
      }
    } else {
      const int lane = threadIdx.x % 32;
      const int global_row = row * WarpM + (threadIdx.x / 32) % WarpM;
      const auto* src = reinterpret_cast<const cute::uint128_t*>(
          shared + plane * Plane + (global_row * 16 + lane % 16) * Stride +
          k * 16 + (lane / 16) * 8);
      cute::SM75_U32x4_LDSM_N::copy(*src, a[0], a[1], a[2], a[3]);
    }
  }

  __device__ __forceinline__ Tile product(const Tile& a, const Tile& b) const {
    stage(b.d, 0);
    if constexpr (Pair)
      stage(b.u, 1);
    if constexpr (!DirectA) {
      stage(a.d, Pair ? 2 : 1);
      if constexpr (Pair)
        stage(a.u, 3);
    }
    sync();
    Tile result;
    constexpr int Columns = !Pair && C >= 2 ? 2 : 1;
    #pragma unroll
    for (int col = 0; col < C; col += Columns) {
      float d[M][Columns][4] = {};
      float du[M][Columns][4] = {}, ud[M][Columns][4] = {};
      #pragma unroll
      for (int k = 0; k < N / 16; ++k) {
        uint32_t bd[2 * Columns], b_upper[2 * Columns];
        const int global_col = col * WarpN + (threadIdx.x / 32) / WarpM;
        if constexpr (Columns == 2)
          load_b_pair(bd, k, global_col, 0);
        else
          load_b(bd, k, global_col, 0);
        if constexpr (Pair)
          load_b(b_upper, k, global_col, 1);
        #pragma unroll
        for (int row = 0; row < M; ++row) {
          uint32_t ad[4], au[4];
          load_a(ad, a.d, row, k, Pair ? 2 : 1);
          if constexpr (Pair)
            load_a(au, a.u, row, k, 3);
          #pragma unroll
          for (int c = 0; c < Columns; ++c)
            mma16816(d[row][c], ad, bd + 2 * c);
          if constexpr (Pair) {
            mma16816(du[row][0], ad, b_upper);
            mma16816(ud[row][0], au, bd);
          }
        }
      }
      #pragma unroll
      for (int row = 0; row < M; ++row) {
        #pragma unroll
        for (int c = 0; c < Columns; ++c) {
          #pragma unroll
          for (int j = 0; j < 4; ++j) {
            const int e = (row * C + col + c) * 4 + j;
            result.d[e] = d[row][c][j];
            if constexpr (Pair)
              result.u[e] = du[row][c][j] + ud[row][c][j];
          }
        }
      }
    }
    sync();
    return result;
  }

  __device__ __forceinline__ Tile add(const Tile& a, const Tile& b) const {
    Tile c;
    #pragma unroll
    for (int e = 0; e < Elements; ++e) {
      c.d[e] = a.d[e] + b.d[e];
      if constexpr (Pair)
        c.u[e] = a.u[e] + b.u[e];
    }
    return c;
  }

  template <int Kind>
  __device__ __forceinline__ Tile combine(
      const Tile& a,
      const Tile& a2,
      const Tile& a3,
      const Tile& a6,
      float c0,
      float c1,
      float c2,
      float c3,
      float c6) const {
    Tile b;
    #pragma unroll
    for (int e = 0; e < Elements; ++e) {
      const int i = Ops::index(e);
      if constexpr (Kind == 5)
        b.d[e] = c2 * a2.d[e];
      else if constexpr (Kind == 3 || Kind == 4)
        b.d[e] = c0 * float(i / N == i % N) + c1 * a.d[e];
      else
        b.d[e] = c1 * a.d[e];
      if constexpr (Kind != 5)
        b.d[e] += c2 * a2.d[e];
      b.d[e] += c3 * a3.d[e];
      if constexpr (Kind != 1)
        b.d[e] += c6 * a6.d[e];
      if constexpr (Pair) {
        if constexpr (Kind == 5)
          b.u[e] = c2 * a2.u[e];
        else
          b.u[e] = c1 * a.u[e];
        if constexpr (Kind != 5)
          b.u[e] += c2 * a2.u[e];
        b.u[e] += c3 * a3.u[e];
        if constexpr (Kind != 1)
          b.u[e] += c6 * a6.u[e];
      }
    }
    return b;
  }

  __device__ __forceinline__ Tile t18(const Tile& a) const {
    auto a2 = product(a, a);
    auto a3 = product(a2, a);
    auto a6 = product(a3, a3);
    auto b1 = combine<1>(
        a, a2, a3, a6, 0.f, -0.10036558103014462f,
        -0.008029246482411570f, -0.000892138498045730f, 0.f);
    auto b2 = combine<2>(
        a, a2, a3, a6, 0.f, 0.39784974949645076f,
        1.36783778460411719f, 0.49828962252538268f, -0.00063789819459247233f);
    auto b3 = combine<3>(
        a, a2, a3, a6, -10.96763960529620626f, 1.68015813878906197f,
        0.05717798464788655f, -0.00698210122488052f, 3.349750170860705e-05f);
    auto b4 = combine<4>(
        a, a2, a3, a6, -0.09043168323908106f, -0.06764045190713819f,
        0.06759613017740597f, 0.02955525704293155f, -1.391802575160607e-05f);
    auto b5 = combine<5>(
        a, a2, a3, a6, 0.f, 0.f, -0.09233646193671186f,
        -0.01693649390020817f, -1.400867981820361e-05f);
    auto a9 = add(product(b1, b5), b4);
    return add(b2, product(add(b3, a9), a9));
  }
};

__device__ __forceinline__ void store_value(
    void* out, int dtype, int64_t i, float v) {
  if (dtype == 0)
    static_cast<float*>(out)[i] = v;
  else if (dtype == 1)
    static_cast<c10::BFloat16*>(out)[i] = c10::BFloat16(v);
  else
    static_cast<c10::Half*>(out)[i] = c10::Half(v);
}

template <int N, int Threads, bool Pair, bool FixedTwo>
__global__ __launch_bounds__(Threads) void mma_kernel(
    const float* x,
    const float* g,
    void* out,
    int dtype,
    bool adaptive,
    int requested_s) {
  using Ops = MmaOps<N, Threads / 32, Pair>;
  __shared__ __align__(16) uint16_t storage[Ops::SharedElements];
  __shared__ float operands[(Pair ? 2 : 1) * N * N];
  __shared__ double column_norm[N];
  __shared__ int shared_s;
  Ops ops{storage};
  typename Ops::Tile a;
  const int64_t base = int64_t(blockIdx.x) * N * N;
  bool bad = false;
  #pragma unroll
  for (int e = 0; e < Ops::Elements; ++e) {
    const int i = Ops::index(e);
    a.d[e] = x[base + i];
    bad |= !::isfinite(a.d[e]);
    if constexpr (Pair) {
      a.u[e] = g[base + i];
      bad |= !::isfinite(a.u[e]);
    }
  }
  if (__syncthreads_or(bad)) {
    #pragma unroll
    for (int e = 0; e < Ops::Elements; ++e)
      store_value(out, dtype, base + Ops::index(e), NAN);
    return;
  }
  int s = 2;
  if constexpr (!FixedTwo) {
    s = requested_s;
    if (adaptive) {
      #pragma unroll
      for (int e = 0; e < Ops::Elements; ++e) {
        const int i = Ops::index(e);
        operands[i] = a.d[e];
        if constexpr (Pair)
          operands[N * N + i] = a.u[e];
      }
      __syncthreads();
      if (threadIdx.x < N) {
        double v = 0.;
        #pragma unroll
        for (int row = 0; row < N; ++row) {
          v += ::fabs(double(operands[row * N + threadIdx.x]));
          if constexpr (Pair)
            v += ::fabs(double(operands[N * N + row * N + threadIdx.x]));
        }
        column_norm[threadIdx.x] = v;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        double norm = 0.;
        for (int col = 0; col < N; ++col)
          norm = ::fmax(norm, column_norm[col]);
        shared_s = norm > 3.010066362817634
            ? int(::ceil(::log2(norm / 3.010066362817634)))
            : 0;
      }
      __syncthreads();
      s = shared_s;
    }
  }
  const float scale = ::ldexpf(1.f, -s);
  #pragma unroll
  for (int e = 0; e < Ops::Elements; ++e) {
    a.d[e] *= scale;
    if constexpr (Pair)
      a.u[e] *= scale;
  }
  auto result = ops.t18(a);
  if constexpr (FixedTwo) {
    #pragma unroll
    for (int i = 0; i < 2; ++i)
      result = ops.product(result, result);
  } else {
    for (int i = 0; i < s; ++i)
      result = ops.product(result, result);
  }
  #pragma unroll
  for (int e = 0; e < Ops::Elements; ++e)
    store_value(out, dtype, base + Ops::index(e), Pair ? result.u[e] : result.d[e]);
}

#else
// PTX compiled for an older target keeps a functional scalar fallback.
template <int N, int Threads, bool Pair, bool FixedTwo>
__global__ __launch_bounds__(Threads) void mma_kernel(
    const float* x,
    const float* g,
    void* out,
    int dtype,
    bool adaptive,
    int requested_s) {
  scalar::compute<N, Threads, Pair, FixedTwo>(x, g, out, dtype, adaptive, requested_s);
}
#endif
} // namespace at::native::small_matrix_exp::mma
