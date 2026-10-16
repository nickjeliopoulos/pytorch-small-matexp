from pathlib import Path

from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NATIVE = REPO / "aten/src/ATen/native"
ARTIFACTS = HERE / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
overlay = ARTIFACTS / "include"
for name in (
    "SmallMatrixExp.h",
    "cuda/MatrixExpSmall.cuh",
):
    link = overlay / "ATen/native" / name
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.is_symlink():
        link.symlink_to(NATIVE / name)

# Compile the exact edited derivative and its helper without rebuilding libtorch.
manual = (REPO / "torch/csrc/autograd/FunctionsManual.cpp").read_text()
start = manual.index(
    "template <typename func_t>\nstatic Tensor differential_analytic_matrix_function("
)
end = manual.index(
    "// Differential of the symmetric/Hermitian matrix square root", start
)
manual_probe = ARTIFACTS / "manual.cpp"
manual_probe.write_text(
    "#include <ATen/ATen.h>\n#include <ATen/TensorSubclassLikeUtils.h>\n"
    "#include <ATen/core/grad_mode.h>\n"
    '#include "SmallMatrixExp.h"\nnamespace matrix_exp_aten_probe {\n'
    "using at::Tensor;\nusing at::areAnyTensorSubclassLike;\n"
    + manual[start:end]
    + "\n}\n"
)

# Exercise the actual size guard and trivial cases from the public entry point.
# The stock fallback is file-local, so call the wheel's native entry there.
linear = (NATIVE / "LinearAlgebra.cpp").read_text()
start = linear.index("Tensor linalg_matrix_exp(const Tensor& a) {")
end = linear.index("\n// Alias", start)
entry = linear[start:end]
if entry.count("at::native::mexp(a)") != 1:
    raise RuntimeError("Update the probe extraction for the matrix_exp fallback")
entry_probe = ARTIFACTS / "entry.cpp"
entry_probe.write_text(
    "#include <ATen/ATen.h>\n"
    "#include <ATen/detail/MPSHooksInterface.h>\n"
    "#include <ATen/native/LinearAlgebraUtils.h>\n"
    "#include <ATen/ops/linalg_matrix_exp_native.h>\n"
    '#include "SmallMatrixExp.h"\nnamespace matrix_exp_aten_probe {\n'
    "using namespace at;\n"
    "using at::native::squareCheckInputs;\n"
    "using at::native::checkFloatingOrComplex;\n"
    "using at::native::use_small_matrix_exp;\n"
    "using at::native::small_matrix_exp_stub;\n"
    + entry.replace("at::native::mexp(a)", "at::native::linalg_matrix_exp(a)")
    + "\n}\n"
)

setup(
    name="matrix-exp-aten-probe",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            "_matrix_exp_aten_probe",
            [
                str(HERE / "register.cpp"),
                str(manual_probe),
                str(entry_probe),
                str(NATIVE / "cuda/MatrixExpKernel.cu"),
            ],
            include_dirs=[str(overlay), str(NATIVE)],
            # Match DispatchStubImpl's ABI in the x86-64 release wheel.
            define_macros=[
                ("HAVE_AVX2_CPU_DEFINITION", "1"),
                ("HAVE_AVX512_CPU_DEFINITION", "1"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-lineinfo",
                    "--keep",
                    "--keep-dir",
                    str(ARTIFACTS),
                    "-Xptxas=-v",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
