.PHONY: build build-m4ri sync clean

PROJECT_ROOT := $(shell pwd)

M4RI_SRC := $(PROJECT_ROOT)/src/scloop/utils/linear_algebra_gf2/m4ri
M4RI_PREFIX := $(PROJECT_ROOT)/src/scloop/utils/linear_algebra_gf2
DM_PREFIX := $(PROJECT_ROOT)/src/scloop/utils/distance_metrics

build-m4ri:
	cd $(M4RI_SRC) && \
		autoreconf -i && \
		./configure --prefix=$(M4RI_PREFIX) --libdir=$(M4RI_PREFIX) \
					--enable-static --disable-shared --enable-openmp \
					CFLAGS="-fPIC -O3" && \
		$(MAKE) && \
		$(MAKE) install

build: build-m4ri
	CFLAGS="-I$(M4RI_PREFIX)/include" \
	LDFLAGS="-L$(M4RI_PREFIX) -Wl,-rpath,$(M4RI_PREFIX)" \
	CPLUS_INCLUDE_PATH=$(PROJECT_ROOT)/src/scloop/data:$(DM_PREFIX) \
		uv build

fresh-sync: build-m4ri
	CPLUS_INCLUDE_PATH=$(PROJECT_ROOT)/src/scloop/data:$(DM_PREFIX)/discrete-frechet-distance uv sync

sync: clean build-m4ri
	python setup.py build_ext --inplace
	CPLUS_INCLUDE_PATH=$(PROJECT_ROOT)/src/scloop/data:$(DM_PREFIX)/discrete-frechet-distance uv sync

full-sync: build-m4ri
	uv cache clean
	uv cache prune
	rm -rf .venv
	rm -rf ./src/scloop.egg-info
	rm -f uv.lock
	find ./src -name '*.so'  -type f -delete
	CPLUS_INCLUDE_PATH=$(PROJECT_ROOT)/src/scloop/data:$(DM_PREFIX)/discrete-frechet-distance uv sync

clean:
	rm -rf dist/ *.egg-info
	cd $(M4RI_SRC) && $(MAKE) clean || true
	rm -rf $(M4RI_PREFIX)/include $(M4RI_PREFIX)/lib*.a $(M4RI_PREFIX)/lib*.so* $(M4RI_PREFIX)/lib*.la $(M4RI_PREFIX)/pkgconfig
	rm -f $(DM_PREFIX)/*.so

rebuild: clean build
