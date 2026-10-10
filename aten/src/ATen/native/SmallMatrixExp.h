#pragma once

#include <ATen/Context.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>

namespace at::native {

inline bool use_small_matrix_exp(const Tensor& a) {
  const auto dtype = a.scalar_type();
  return a.is_cuda() && a.layout() == kStrided && a.dim() == 3 &&
      a.size(-1) == a.size(-2) && a.size(-1) > 1 && a.size(-1) <= 32 &&
      (dtype == kFloat || dtype == kHalf || dtype == kBFloat16) &&
      !at::globalContext().hasROCM();
}

using small_matrix_exp_fn = Tensor (*)(const Tensor&, const std::optional<Tensor>&);
DECLARE_DISPATCH(small_matrix_exp_fn, small_matrix_exp_stub)

} // namespace at::native
