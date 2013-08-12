NAME=libhalloc.a
SRC_C=src/*.cu
SRC_H=src/*.h src/*.cuh
SRC=$(SRC_C) $(SRC_H)
TGT=bin/$(NAME)

#TEST_TGT=tst/corr/bin/test

TMP=*~ \\\#* src/*~ src/\\\#* tst/corr/*~ tst/corr/*.o $(TGT) $(TEST_TGT)

build: $(TGT)
$(TGT):	$(SRC) makefile
	nvcc -arch=sm_35 -O3 -lib -rdc=true -o $(TGT) $(SRC_C)

test: $(TGT) makefile
	make -C tst/corr/test run

clean:
	rm -f $(TMP)
