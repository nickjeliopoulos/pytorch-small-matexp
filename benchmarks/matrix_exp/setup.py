from pathlib import Path

from setuptools import find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

setup(
    name="matrix-exp-bench",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            "matrix_exp_bench._C",
            [
                str(ROOT / "csrc" / name)
                for name in ("register.cpp", "rows.cu", "scalar.cu", "mma.cu")
            ],
            define_macros=[("MATEXP_CORE_ONLY", "1")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-lineinfo",
                    "--keep",
                    "--keep-dir",
                    str(ARTIFACTS),
                    "-Xptxas=-v",
                    "--expt-relaxed-constexpr",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
