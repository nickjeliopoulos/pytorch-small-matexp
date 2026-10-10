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
    "cuda/MatrixExpRows.cuh",
    "cuda/MatrixExpScalar.cuh",
    "cuda/MatrixExpMma.cuh",
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

setup(
    name="matrix-exp-aten-probe",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            "_matrix_exp_aten_probe",
            [
                str(HERE / "register.cpp"),
                str(manual_probe),
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
