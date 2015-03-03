include ../../common.mk
include ../common-def.mk

TGT=libcommontest.a
SRC_C=*.cu
SRC_H=*.h
SRC=$(SRC_C) $(SRC_H)
TMP=$(TGT) *~ \#*

FLAGS= $(NVCC_ARCH) -O3 -rdc=true -lib -I../include
ifeq ($(WITH_SCATTER),yes)
FLAGS += -DWITH_SCATTER
endif

build: $(TGT) 
$(TGT):	$(SRC) makefile
	nvcc $(FLAGS) -o $(TGT) $(SRC_C)

clean:
	rm -f $(TMP)
