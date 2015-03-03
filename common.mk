NVCC_ARCH:= -gencode arch=compute_20,code=sm_20 \
	-gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_35,code=sm_35

NVCC_MAJOR_VERSION:=$(shell nvcc --version | grep -oE "release [0-9]+" | grep -oE "[0-9]+")

#$(info $(NVCC_MAJOR_VERSION))

MAXWELL_SUPPORTED:=$(shell if [ $(NVCC_MAJOR_VERSION) -ge 6 ]; then echo yes; \
	else echo no; fi)

#$(info $(MAXWELL_SUPPORTED))

ifeq ($(MAXWELL_SUPPORTED),yes)
	NVCC_ARCH += -gencode arch=compute_50,code=sm_50
endif
