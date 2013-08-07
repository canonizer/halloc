NAME=libhalloc.a
SRC_C=src/*.cu
SRC_H=src/*.h src/*.cuh
SRC=$(SRC_C) $(SRC_H)
TGT=bin/$(NAME)

TEST_SRC=tst/corr/test.cu
TEST_TGT=tst/corr/bin/test
TEST_OBJ=tst/corr/test.o

TMP=*~ \\\#* src/*~ src/\\\#* tst/corr/*~ tst/corr/*.o $(TGT) $(TEST_TGT)

build: $(TGT)
$(TGT):	$(SRC) makefile
	nvcc -arch=sm_35 -O3 -lib -rdc=true -o $(TGT) $(SRC_C)

$(TEST_TGT): $(TEST_OBJ) $(TGT) makefile
	nvcc -arch=sm_35 -O3 -Xcompiler -fopenmp $(TEST_OBJ) $(TGT) -o $(TEST_TGT)

$(TEST_OBJ): $(TEST_SRC) makefile
	nvcc -arch=sm_35 -O3 -Xcompiler -fopenmp -Itst/include -dc $(TEST_SRC) -o \
		$(TEST_OBJ)


test: $(TEST_TGT)
	./$(TEST_TGT)

clean:
	rm -f $(TMP)
