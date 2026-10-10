#include <ATen/ATen.h>
#include <torch/library.h>

void matrix_exp_out(const at::Tensor&, const std::optional<at::Tensor>&,
                    const at::Tensor&, bool, int64_t);
void tiled_out(const at::Tensor&, const std::optional<at::Tensor>&,
               const at::Tensor&, bool, int64_t, int64_t);
void mma_out(const at::Tensor&, const std::optional<at::Tensor>&,
             const at::Tensor&, bool, int64_t, int64_t);

TORCH_LIBRARY(matrix_exp_bench, m) {
  m.def("rows_out(Tensor X, Tensor? G, Tensor(a!) out, bool adaptive, int S) -> ()");
  m.def("scalar_out(Tensor X, Tensor? G, Tensor(a!) out, bool adaptive, int S, int threads) -> ()");
  m.def("mma_out(Tensor X, Tensor? G, Tensor(a!) out, bool adaptive, int S, int threads) -> ()");
}

TORCH_LIBRARY_IMPL(matrix_exp_bench, CUDA, m) {
  m.impl("rows_out", matrix_exp_out);
  m.impl("scalar_out", tiled_out);
  m.impl("mma_out", mma_out);
}
