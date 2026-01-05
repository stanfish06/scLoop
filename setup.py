import os
import sys

from Cython.Build import cythonize
from setuptools import Extension, setup

project_root = os.path.dirname(os.path.abspath(__file__))
gf2_dir = os.path.join(project_root, "src/scloop/utils/linear_algebra_gf2")
gf2toolkit_srcs = os.path.join(gf2_dir, "GF2toolkit/srcs")

is_windows = sys.platform.startswith("win")
if is_windows:
    base_link_args = ["-static-libgcc", "-static-libstdc++", "-Wl,-Bstatic", "-lpthread", "-Wl,-Bdynamic"]
    openmp_compile = ["-fopenmp"]
    openmp_link = base_link_args + ["-fopenmp", "-Wl,-Bstatic", "-lgomp", "-Wl,-Bdynamic"]
else:
    base_link_args = []
    openmp_compile = ["-fopenmp"]
    openmp_link = ["-fopenmp"]

extensions = [
    Extension(
        "scloop.data.ripser_lib",
        sources=["./src/scloop/data/ripser_lib.pyx", "./src/scloop/data/ripser.cpp"],
        language="c++",
        include_dirs=["./src/scloop/data"],
        extra_link_args=base_link_args,
    ),
    Extension(
        "scloop.utils.linear_algebra_gf2.m4ri_lib",
        sources=["./src/scloop/utils/linear_algebra_gf2/m4ri_lib.pyx"],
        include_dirs=[os.path.join(gf2_dir, "include")],
        extra_objects=[os.path.join(gf2_dir, "libm4ri.a")],
        extra_compile_args=openmp_compile,
        extra_link_args=openmp_link,
    ),
    # Extension(
    #     "scloop.utils.linear_algebra_gf2.gf2toolkit_lib",
    #     sources=[
    #         "./src/scloop/utils/linear_algebra_gf2/gf2toolkit_lib.pyx",
    #         "./src/scloop/utils/linear_algebra_gf2/gf2toolkit_wrapper.cpp",
    #     ],
    #     include_dirs=[
    #         gf2_dir,
    #         gf2toolkit_srcs,
    #         os.path.join(gf2_dir, "include"),
    #         os.path.join(gf2_dir, "GF2toolkit/submodules/m4ri"),
    #     ],
    #     extra_objects=[
    #         os.path.join(gf2_dir, "libGF2toolkit.a"),
    #         os.path.join(gf2_dir, "libm4ri.a"),
    #     ],
    #     language="c++",
    #     extra_compile_args=["-std=c++11", "-O3", "-fopenmp"],
    #     extra_link_args=["-fopenmp"],
    # ),
    Extension(
        "scloop.utils.distance_metrics.frechet",
        sources=[
            "./src/scloop/utils/distance_metrics/frechet.pyx",
            "./src/scloop/utils/distance_metrics/discrete-frechet-distance/Frechet.cpp",
        ],
        include_dirs=["./src/scloop/utils/distance_metrics/discrete-frechet-distance"],
        language="c++",
        extra_link_args=base_link_args,
    ),
]

setup(ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}))
