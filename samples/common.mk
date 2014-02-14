LIB_DIR=-L../../bin
#LIB_DIR=-L ~/usr/lib
LIBHALLOC=-lhalloc
LIBHALLOC_FILE=../../bin/libhalloc.a
INCLUDE_DIR=-I../include
#INCLUDE_DIR=-I ~/usr/include
SRC_C=*.cu
SRC_H=../include/halloc.h
SRC=$(SRC_C) $(SRC_H)
#SRC=$(SRC_C)
TGT=../bin/$(NAME)

OBJ=../tmp/$(NAME).o

TMP=*~ \\\#* ../tmp/*.o $(TGT)

build: $(TGT)
$(TGT): $(LIBHALLOC_FILE) $(OBJ) makefile
#	nvcc -arch=sm_35 -O3 -Xcompiler -fopenmp $(OBJ) $(LIBHALLOC) -o $(TGT)
	nvcc -arch=sm_35 -O3 -Xcompiler -fopenmp $(LIB_DIR) $(LIBHALLOC) -o \
	  $(TGT) $(OBJ)

$(OBJ): $(SRC) makefile
	nvcc -arch=sm_35 -O3 -Xcompiler -fopenmp -Xptxas -dlcm=cg -Xptxas -dscm=wb \
		-Xcompiler -pthread $(INCLUDE_DIR) -dc $(SRC_C) -o $(OBJ)

run: $(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)

$(LIBHALLOC):
	make -C ../..
