.PHONY: build build-m4ri sync clean

PROJECT_ROOT := $(shell pwd)

M4RI_SRC := $(PROJECT_ROOT)/src/scloop/utils/linear_algebra_gf2/m4ri
M4RI_PREFIX := $(PROJECT_ROOT)/src/scloop/utils/linear_algebra_gf2

build-m4ri:
	cd $(M4RI_SRC) && \
		autoreconf -i && \
		./configure --prefix=$(M4RI_PREFIX) --libdir=$(M4RI_PREFIX) --disable-static --enable-shared --enable-openmp && \
		$(MAKE) && \
		$(MAKE) install

build: build-m4ri
	CPLUS_INCLUDE_PATH=$(PROJECT_ROOT)/src/scloop/data uv build

sync:
	CPLUS_INCLUDE_PATH=$(PROJECT_ROOT)/src/scloop/data uv sync

clean:
	rm -rf dist/ *.egg-info
	cd $(M4RI_SRC) && $(MAKE) clean || true
	rm -rf $(M4RI_PREFIX)/include $(M4RI_PREFIX)/lib*.so* $(M4RI_PREFIX)/lib*.la $(M4RI_PREFIX)/pkgconfig

rebuild: clean build
