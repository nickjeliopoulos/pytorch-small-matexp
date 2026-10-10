#include "SmallMatrixExp.h"
#include <ATen/ops/linalg_matrix_exp_native.h>
#include <torch/library.h>

namespace at::native {
DEFINE_DISPATCH(small_matrix_exp_stub);
REGISTER_NO_CPU_DISPATCH(small_matrix_exp_stub);
Tensor small_matrix_exp_cuda(const Tensor&, const std::optional<Tensor>&);
} // namespace at::native

namespace matrix_exp_aten_probe {
at::Tensor linalg_matrix_exp_differential(const at::Tensor&, const at::Tensor&, bool);
}

namespace {
at::Tensor forward(const at::Tensor& a) {
  return at::native::use_small_matrix_exp(a)
      ? at::native::small_matrix_exp_stub(at::kCUDA, a, std::nullopt)
      : at::native::linalg_matrix_exp(a);
}

at::Tensor frechet(const at::Tensor& a, const at::Tensor& g) {
  return at::native::small_matrix_exp_cuda(a, g);
}
} // namespace

TORCH_LIBRARY(matrix_exp_aten_probe, m) {
  m.def("forward(Tensor a) -> Tensor");
  m.def("frechet(Tensor a, Tensor g) -> Tensor");
  m.def("eligible(Tensor a) -> bool");
  m.def("differential(Tensor a, Tensor g, bool adjoint) -> Tensor");
}

TORCH_LIBRARY_IMPL(matrix_exp_aten_probe, CompositeImplicitAutograd, m) {
  m.impl("forward", &forward);
  m.impl("frechet", &frechet);
  m.impl("eligible", &at::native::use_small_matrix_exp);
  m.impl("differential", &matrix_exp_aten_probe::linalg_matrix_exp_differential);
}
