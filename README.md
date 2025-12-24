# scloop - single-cell loop analysis

[![Test Import](https://github.com/stanfish06/scLoop/actions/workflows/test-import.yml/badge.svg)](https://github.com/stanfish06/scLoop/actions/workflows/test-import.yml)
[![Test Build](https://github.com/stanfish06/scLoop/actions/workflows/test-build.yml/badge.svg)](https://github.com/stanfish06/scLoop/actions/workflows/test-build.yml)

**Note: this package is still under development.**

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
**Note: this package is still under development. The current build will likely not work.**
```bash
pip install scloop
```

## Project Structure
```
src/scloop/
├── analyzing
│   └── __init__.py
├── benchmarking
│   └── __init__.py
├── computing
│   ├── homology.py
│   └── __init__.py
├── data
│   ├── analysis_containers.py
│   ├── base_components.py
│   ├── containers.py
│   ├── __init__.py
│   ├── loop_reconstruction.py
│   ├── metadata.py
│   ├── ripser.cpp
│   ├── ripser.hpp
│   ├── ripser_lib.cpp
│   ├── ripser_lib.pyx
│   ├── types.py
│   └── utils.py
├── __init__.py
├── matching
│   ├── cross_dataset.py
│   ├── data_modules.py
│   ├── __init__.py
│   ├── mlp.py
│   └── nf.py
├── plotting
│   ├── _hodge.py
│   ├── _homology.py
│   ├── __init__.py
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
    │   ├── frechet.pyx
    │   └── __init__.py
    ├── __init__.py
    └── linear_algebra_gf2
        ├── gf2_toolkit_lib.pyx
        ├── __init__.py
        ├── m4ri_lib.c
        └── m4ri_lib.pyx
```
