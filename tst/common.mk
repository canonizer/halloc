LIBHALLOC=../../../bin/libhalloc.a
LIBCOMMON=../../common/libcommontest.a
LIBSCATTER=../../include/libscatteralloc.a
LIBS=$(LIBHALLOC) $(LIBCOMMON) $(LIBSCATTER)
SRC_C=*.cu
SRC_H=../../include/halloc.h ../../common/*.h
SRC=$(SRC_C) $(SRC_H)
#SRC=$(SRC_C)
TGT=../bin/$(NAME)

OBJ=../tmp/$(NAME).o

TMP=*~ \\\#* ../tmp/*.o $(TGT)

build: $(TGT)
$(TGT): $(LIBS) $(OBJ) makefile
	nvcc -arch=sm_35 -O3 -Xcompiler -fopenmp $(OBJ) $(LIBS) -o $(TGT)

$(OBJ): $(SRC) makefile
	nvcc -arch=sm_35 -O3 -Xcompiler -fopenmp -I../../include -I../../common \
	  -dc $(SRC_C) -o $(OBJ)

run: $(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
