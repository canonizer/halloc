LIBHALLOC=../../bin/libhalloc.a
SRC_C=*.cu
SRC_H=../include/halloc.h
SRC=$(SRC_C) $(SRC_H)
#SRC=$(SRC_C)
TGT=../bin/$(NAME)

OBJ=../tmp/$(NAME).o

TMP=*~ \\\#* ../tmp/*.o $(TGT)

build: $(TGT)
$(TGT): $(LIBHALLOC) $(OBJ) makefile
	nvcc -arch=sm_35 -O3 -Xcompiler -fopenmp $(OBJ) $(LIBHALLOC) -o $(TGT)

$(OBJ): $(SRC) makefile
	nvcc -arch=sm_35 -O3 -Xcompiler -fopenmp -I../include -dc $(SRC_C) -o $(OBJ)

run: $(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
