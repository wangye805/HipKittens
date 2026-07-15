# DSA-V4 sparse-MLA forward (occ-2, gfx950). Mirrors kernels/attn/gqa.
GPU_TARGET=CDNA4
TARGET=tk_kernel

# ---- shape / config knobs ----
TOPK   ?= 1152          # number of gathered KV rows per query token
DC     ?= 64            # D-channel chunk (64 = occ-2 sweet spot; 128 single-buf ties, 256 spills)
HAS_SINK ?= 1
TILE_K := 32
NTILES := $(shell expr $(TOPK) / $(TILE_K))

CPPFLAGS += -DNTILES=$(NTILES) -DDC=$(DC) -DHAS_SINK=$(HAS_SINK)

SRC=kernel.cpp

# ---- HIP ----
ROCM_PATH ?= /opt/rocm
ROCM_INSTALL_DIR := $(ROCM_PATH)
HIP_INCLUDE_DIR  := $(ROCM_INSTALL_DIR)/include/hip
HIPCXX ?= $(ROCM_INSTALL_DIR)/bin/hipcc
THUNDERKITTENS_ROOT ?= $(abspath ../../..)

# CDNA4 (gfx950) + the DSA fwd perf flags (matched-VGPR MFMA form + permlane reduce).
HIPFLAGS += -DKITTENS_CDNA4 --offload-arch=gfx950 -DHIP_ENABLE_WARP_SYNC_BUILTINS -ffast-math
HIPFLAGS += -mllvm -amdgpu-mfma-vgpr-form -DHK_PERMLANE_REDUCE

CXX_STD   := c++20
ICXXFLAGS := -std=$(CXX_STD) -w
ICPPFLAGS := -I$(THUNDERKITTENS_ROOT)/include -I$(THUNDERKITTENS_ROOT)/prototype -I$(HIP_INCLUDE_DIR)

PY_LDFLAGS := $(shell python3-config --ldflags | sed 's/-lcrypt//g')
ICXXFLAGS += $(PY_LDFLAGS) $(shell python3 -m pybind11 --includes) -shared -fPIC

all: $(TARGET)

$(TARGET): $(SRC)
	$(HIPCXX) $(SRC) $(HIPFLAGS) $(ICXXFLAGS) $(ICPPFLAGS) $(CPPFLAGS) \
	    -o $(TARGET)$(shell python3 -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")

# C++ correctness (no python): multi-token batched test vs bf16-faithful CPU reference.
test_cpp: test_batched.cpp kernel.cpp
	$(HIPCXX) test_batched.cpp $(HIPFLAGS) -std=$(CXX_STD) -w -O3 $(ICPPFLAGS) $(CPPFLAGS) -o test_batched
	./test_batched 8 0 0

clean:
	rm -f $(TARGET)*.so test_batched
