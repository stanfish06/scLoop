# scloop - single-cell loop analysis

[![Test Import](https://github.com/stanfish06/scLoop/actions/workflows/test-import.yml/badge.svg)](https://github.com/stanfish06/scLoop/actions/workflows/test-import.yml)
[![Test Build](https://github.com/stanfish06/scLoop/actions/workflows/test-build.yml/badge.svg)](https://github.com/stanfish06/scLoop/actions/workflows/test-build.yml)
[![PyPI](https://img.shields.io/pypi/v/scloop)](https://pypi.org/project/scloop)

**Note: this package is in early stages of development.**

scLoop is a library to identify statistically significant loops in single-cell RNA-seq data.

![persistent homology](https://raw.githubusercontent.com/stanfish06/scloop/master/examples/loop.png)

## Basic Usage
```python
import scloop as scl
scl.pp.prepare_adata(adata, downsample=True, n_downsample=500)
scl.tl.find_loops(adata, ...)
scl.tl.analyze_loops(adata, ...)
scl.tl.match_loops([adata1, adata2, ...], ...)
```

## Build Instructions
```bash
make build
```

or

```bash
make rebuild
```

## Virtual Envrionemnt
```bash
make sync
```

or

```bash
make full-sync
```

to prevent partial compilation of some modules

## Installation
**Note: this package is in early stages of development. The current build will have issues.**
```bash
pip install scloop
```

## Project Structure
```
src/scloop/
├── analyzing
│   ├── bootstrap.py
│   ├── feature_selection.py
│   ├── gene_trend.py
│   ├── hodge.py
│   ├── __init__.py
│   └── stats.py
├── benchmarking
│   └── __init__.py
├── computing
│   ├── boundary.py
│   ├── hodge_decomposition.py
│   ├── homology.py
│   ├── __init__.py
│   ├── loops.py
│   └── matching.py
├── data
│   ├── analysis_containers.py
│   ├── base_components.py
│   ├── boundary.py
│   ├── constants.py
│   ├── containers.py
│   ├── __init__.py
│   ├── metadata.py
│   ├── ripser.cpp
│   ├── ripser.hpp
│   ├── ripser_lib.cpp
│   ├── ripser_lib.pyx
│   ├── types.py
│   └── utils.py
├── __init__.py
├── io
│   └── __init__.py
├── matching
│   ├── cross_dataset.py
│   ├── data_modules.py
│   ├── __init__.py
│   ├── mlp.py
│   └── nf.py
├── plotting
│   ├── _cross_match.py
│   ├── custom_colormaps.py
│   ├── _hodge.py
│   ├── _homology.py
│   ├── __init__.py
│   ├── _trajectory.py
│   └── _utils.py
├── preprocessing
│   ├── delve
│   │   ├── delve.py
│   │   ├── __init__.py
│   │   └── kh.py
│   ├── downsample.py
│   ├── __init__.py
│   └── prepare.py
├── py.typed
├── tools
│   ├── _cross_match.py
│   ├── __init__.py
│   └── _loops.py
└── utils
    ├── distance_metrics
    │   ├── discrete-frechet-distance
    │   ├── frechet.cpp
    │   ├── frechet_py.py
    │   ├── frechet.pyx
    │   └── __init__.py
    ├── __init__.py
    ├── linear_algebra_gf2
    │   ├── gf2_toolkit_lib.pyx
    │   ├── __init__.py
    │   ├── m4ri_lib.c
    │   └── m4ri_lib.pyx
    ├── logging.py
    └── pvalues.py
```
