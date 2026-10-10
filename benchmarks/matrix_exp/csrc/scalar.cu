#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace {
// One CTA per matrix. Each lane owns consecutive row-major tile elements at
// a block-sized stride. Polynomial intermediates remain in registers; only
// operands of the current product are staged in shared memory.
template <int N, int Threads, bool Pair>
struct Cooperative {
  static constexpr int Elements = N*N/Threads;
  static constexpr int Columns = Elements >= 16 ? 4 : Elements >= 2 ? 2 : 1;
  static constexpr int Rows = Elements/Columns;
  using Ops = Cooperative<N,Threads,Pair>;
  __device__ __forceinline__ static int index(int e) {
    const int row = threadIdx.x/(N/Columns) + (e/Columns)*(Threads/(N/Columns));
    const int col = (threadIdx.x%(N/Columns))*Columns + e%Columns;
    return row*N+col;
  }
  struct Tile { float d[Elements]; float u[Pair ? Elements : 1]; };
  float* shared;

  __device__ __forceinline__ Tile product(const Tile& a, const Tile& b) const {
#pragma unroll
    for (int e=0; e<Elements; ++e) {
      const int i = Ops::index(e);
      shared[i] = a.d[e];
      shared[N*N+i] = b.d[e];
      if constexpr (Pair) {
        shared[2*N*N+i] = a.u[e];
        shared[3*N*N+i] = b.u[e];
      }
    }
    __syncthreads();
    Tile result;
    float du[Elements], ud[Elements];
#pragma unroll
    for (int e=0; e<Elements; ++e) {
      result.d[e] = 0.f;
      if constexpr (Pair) { du[e] = 0.f; ud[e] = 0.f; }
    }
    // Reuse each A value across Columns outputs and each B value across Rows.
#pragma unroll
    for (int k=0; k<N; ++k) {
      float bd[Columns], b_upper[Columns];
#pragma unroll
      for (int col=0; col<Columns; ++col) {
        const int j = index(col)%N;
        bd[col] = shared[N*N+k*N+j];
        if constexpr (Pair) b_upper[col] = shared[3*N*N+k*N+j];
      }
#pragma unroll
      for (int row=0; row<Rows; ++row) {
        const int i = index(row*Columns)/N;
        const float ad = shared[i*N+k];
        float au = 0.f;
        if constexpr (Pair) au = shared[2*N*N+i*N+k];
#pragma unroll
        for (int col=0; col<Columns; ++col) {
          const int e = row*Columns+col;
          result.d[e] = fmaf(ad,bd[col],result.d[e]);
          if constexpr (Pair) {
            du[e] = fmaf(ad,b_upper[col],du[e]);
            ud[e] = fmaf(au,bd[col],ud[e]);
          }
        }
      }
    }
    if constexpr (Pair) {
#pragma unroll
      for (int e=0; e<Elements; ++e) result.u[e] = du[e]+ud[e];
    }
    // Every reader must finish before the next product reuses shared storage.
    __syncthreads();
    return result;
  }

  __device__ __forceinline__ Tile add(const Tile& a, const Tile& b) const {
    Tile c;
#pragma unroll
    for (int e=0; e<Elements; ++e) {
      c.d[e] = a.d[e] + b.d[e];
      if constexpr (Pair) c.u[e] = a.u[e] + b.u[e];
    }
    return c;
  }

  __device__ __forceinline__ Tile combine(const Tile& a, const Tile& a2,
      const Tile& a3, const Tile& a6, float c0, float c1, float c2,
      float c3, float c6) const {
    Tile b;
#pragma unroll
    for (int e=0; e<Elements; ++e) {
      const int i = Ops::index(e);
      b.d[e] = c0*float(i/N == i%N) + c1*a.d[e] + c2*a2.d[e]
          + c3*a3.d[e] + c6*a6.d[e];
      if constexpr (Pair)
        b.u[e] = c1*a.u[e] + c2*a2.u[e] + c3*a3.u[e] + c6*a6.u[e];
    }
    return b;
  }

  __device__ __forceinline__ Tile t18(const Tile& a) const {
    auto a2 = product(a,a);
    auto a3 = product(a2,a);
    auto a6 = product(a3,a3);
    auto b1 = combine(a,a2,a3,a6,0.f,-0.10036558103014462f,
        -0.008029246482411570f,-0.000892138498045730f,0.f);
    auto b2 = combine(a,a2,a3,a6,0.f,0.39784974949645076f,
        1.36783778460411719f,0.49828962252538268f,-0.00063789819459247233f);
    auto b3 = combine(a,a2,a3,a6,-10.96763960529620626f,1.68015813878906197f,
        0.05717798464788655f,-0.00698210122488052f,3.349750170860705e-05f);
    auto b4 = combine(a,a2,a3,a6,-0.09043168323908106f,-0.06764045190713819f,
        0.06759613017740597f,0.02955525704293155f,-1.391802575160607e-05f);
    auto b5 = combine(a,a2,a3,a6,0.f,0.f,-0.09233646193671186f,
        -0.01693649390020817f,-1.400867981820361e-05f);
    auto a9 = add(product(b1,b5),b4);
    return add(b2,product(add(b3,a9),a9));
  }
};

__device__ __forceinline__ void store_value(void* out, int dtype, int64_t i, float v) {
  if (dtype == 0) static_cast<float*>(out)[i] = v;
  else if (dtype == 1) static_cast<c10::BFloat16*>(out)[i] = c10::BFloat16(v);
  else static_cast<c10::Half*>(out)[i] = c10::Half(v);
}

template <int N, int Threads, bool Pair, bool FixedTwo>
__global__ __launch_bounds__(Threads) void tiled_kernel(
    const float* x, const float* g, void* out, int dtype, bool adaptive, int requested_s) {
  using Ops = Cooperative<N,Threads,Pair>;
  __shared__ float operands[(Pair ? 4 : 2)*N*N];
  __shared__ double column_norm[N];
  __shared__ int shared_s;
  Ops ops{operands};
  typename Ops::Tile a;
  const int64_t base = int64_t(blockIdx.x)*N*N;
  bool bad = false;
#pragma unroll
  for (int e=0; e<Ops::Elements; ++e) {
    const int i = Ops::index(e);
    a.d[e] = x[base+i];
    bad |= !isfinite(a.d[e]);
    if constexpr (Pair) { a.u[e] = g[base+i]; bad |= !isfinite(a.u[e]); }
  }
  if (__syncthreads_or(bad)) {
#pragma unroll
    for (int e=0; e<Ops::Elements; ++e)
      store_value(out,dtype,base+Ops::index(e),NAN);
    return;
  }
  int s = 2;
  if constexpr (!FixedTwo) {
    s = requested_s;
    if (adaptive) {
#pragma unroll
      for (int e=0; e<Ops::Elements; ++e) {
        const int i = Ops::index(e);
        operands[i] = a.d[e];
        if constexpr (Pair) operands[N*N+i] = a.u[e];
      }
      __syncthreads();
      if (threadIdx.x < N) {
        double v = 0.;
#pragma unroll
        for (int row=0; row<N; ++row) {
          v += fabs(double(operands[row*N+threadIdx.x]));
          if constexpr (Pair) v += fabs(double(operands[N*N+row*N+threadIdx.x]));
        }
        column_norm[threadIdx.x] = v;
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        double norm = 0.;
        for (int col=0; col<N; ++col) norm = fmax(norm,column_norm[col]);
        shared_s = norm > 3.010066362817634 ? int(ceil(log2(norm/3.010066362817634))) : 0;
      }
      __syncthreads();
      s = shared_s;
    }
  }
  const float scale = ldexpf(1.f,-s);
#pragma unroll
  for (int e=0; e<Ops::Elements; ++e) {
    a.d[e] *= scale;
    if constexpr (Pair) a.u[e] *= scale;
  }
  auto result = ops.t18(a);
  if constexpr (FixedTwo) {
#pragma unroll
    for (int i=0; i<2; ++i) result = ops.product(result,result);
  } else {
    for (int i=0; i<s; ++i) result = ops.product(result,result);
  }
#pragma unroll
  for (int e=0; e<Ops::Elements; ++e)
    store_value(out,dtype,base+Ops::index(e),Pair ? result.u[e] : result.d[e]);
}

template<int N,int T,bool Pair>
void launch(const at::Tensor& x, const float* g, const at::Tensor& out,
            int dtype, bool adaptive, int s) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (!adaptive && s == 2)
    tiled_kernel<N,T,Pair,true><<<x.size(0),T,0,stream>>>(x.data_ptr<float>(),g,out.data_ptr(),dtype,adaptive,s);
  else
    tiled_kernel<N,T,Pair,false><<<x.size(0),T,0,stream>>>(x.data_ptr<float>(),g,out.data_ptr(),dtype,adaptive,s);
}

template<int N,bool Pair>
void dispatch(const at::Tensor& x, const float* g, const at::Tensor& out,
              int dtype, bool adaptive, int s, int threads) {
  if (threads == 64) launch<N,64,Pair>(x,g,out,dtype,adaptive,s);
  else if (threads == 128) launch<N,128,Pair>(x,g,out,dtype,adaptive,s);
  else launch<N,256,Pair>(x,g,out,dtype,adaptive,s);
}
} // namespace

void tiled_out(const at::Tensor& x, const std::optional<at::Tensor>& g,
                     const at::Tensor& out, bool adaptive, int64_t s, int64_t threads) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kFloat && x.is_contiguous(), "expected contiguous CUDA FP32 input");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == x.size(2), "expected [B,N,N]");
  TORCH_CHECK(out.device() == x.device() && out.sizes() == x.sizes() && out.is_contiguous(), "invalid output shape/device/layout");
  const int n = x.size(1);
  TORCH_CHECK(n == 16 || n == 32, "cooperative experiment supports N=16/32");
  TORCH_CHECK(threads == 64 || threads == 128 || threads == 256, "threads must be 64/128/256");
  TORCH_CHECK(s >= 0 && s <= 512, "fixed S must be in [0,512]");
  int dtype = out.scalar_type() == at::kFloat ? 0 : out.scalar_type() == at::kBFloat16 ? 1 : out.scalar_type() == at::kHalf ? 2 : -1;
  TORCH_CHECK(dtype >= 0, "invalid output dtype");
  if (g) TORCH_CHECK(g->sizes() == x.sizes() && g->device() == x.device()
      && g->scalar_type() == at::kFloat && g->is_contiguous(), "invalid G");
  if (x.numel() == 0) return;
  const c10::cuda::CUDAGuard guard(x.device());
#define RUN(N) if(g) dispatch<N,true>(x,g->data_ptr<float>(),out,dtype,adaptive,s,threads); else dispatch<N,false>(x,nullptr,out,dtype,adaptive,s,threads)
  if (n == 16) { RUN(16); } else { RUN(32); }
#undef RUN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
