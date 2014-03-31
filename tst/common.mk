include ../../common-def.mk

LIBHALLOC=../../../bin/libhalloc.a
LIBCOMMON=../../common/libcommontest.a
LIBSCATTER=../../include/libscatteralloc.a

LIBS :=$(LIBHALLOC) $(LIBCOMMON)

ARCH := -gencode arch=compute_20,code=sm_20 \
	-gencode arch=compute_30,code=sm_30 \
	-gencode arch=compute_35,code=sm_35

FLAGS := $(ARCH) -O3 -Xcompiler -fopenmp
CUFLAGS := $(FLAGS) -I../../include -I../../common 

ifeq ($(WITH_SCATTER), 1)
LIBS += $(LIBSCATTER)
CUFLAGS += -DWITH_SCATTER
endif

CUFLAGS += -dc

SRC_C=*.cu
SRC_H=../../include/halloc.h ../../common/*.h
SRC=$(SRC_C) $(SRC_H)
TGT=../bin/$(NAME)

OBJ=../tmp/$(NAME).o

TMP=*~ \\\#* ../tmp/*.o $(TGT)

build: $(TGT)
$(TGT): $(LIBS) $(OBJ) makefile
	nvcc $(FLAGS) $(OBJ) $(LIBS) -o $(TGT)

$(OBJ): $(SRC) makefile
	nvcc $(CUFLAGS) -dc $(SRC_C) -o $(OBJ)

run: $(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
