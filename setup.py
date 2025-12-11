import os

from Cython.Build import cythonize
from setuptools import Extension, setup

project_root = os.path.dirname(os.path.abspath(__file__))
m4ri_dir = os.path.join(project_root, "src/scloop/utils/linear_algebra_gf2")

extensions = [
    Extension(
        "scloop.data.ripser_lib",
        sources=["./src/scloop/data/ripser_lib.pyx", "./src/scloop/data/ripser.cpp"],
        language="c++",
        include_dirs=["./src/scloop/data"],
    ),
    Extension(
        "scloop.utils.linear_algebra_gf2.m4ri_lib",
        sources=["./src/scloop/utils/linear_algebra_gf2/m4ri_lib.pyx"],
        include_dirs=[os.path.join(m4ri_dir, "include")],
        extra_objects=[os.path.join(m4ri_dir, "libm4ri.a")],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
    Extension(
        "scloop.utils.distance_metrics.frechet",
        sources=[
            "./src/scloop/utils/distance_metrics/frechet.pyx",
            "./src/scloop/utils/distance_metrics/discrete-frechet-distance/Frechet.cpp",
        ],
        include_dirs=["./src/scloop/utils/distance_metrics/discrete-frechet-distance"],
        language="c++",
    ),
]

setup(ext_modules=cythonize(extensions, compiler_directives={"language_level": "3"}))
