include ../../common-def.mk

LIBHALLOC=../../../bin/libhalloc.a
LIBCOMMON=../../common/libcommontest.a
LIBSCATTER=../../include/libscatteralloc.a

LIBS :=$(LIBHALLOC) $(LIBCOMMON)
FLAGS := -arch=sm_35 -O3 -Xcompiler -fopenmp 
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
