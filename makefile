include common.mk
include makefile.def

NAME=libhalloc.a
HEADER=src/halloc.h
SRC_C=src/*.cu
SRC_H=src/*.h src/*.cuh
SRC=$(SRC_C) $(SRC_H)
TGT=bin/$(NAME)
#TEST_TGT=tst/corr/bin/test

TMP=*~ \\\#* src/*~ src/\\\#* tst/corr/*~ tst/corr/*.o $(TGT) $(TEST_TGT)

# be careful: using -dlcm=cs or ca can lead to errors
build: $(TGT)
$(TGT):	$(SRC) makefile
#	nvcc $(NVCC_ARCH) -lineinfo -O3 -lib -rdc=true -Xptxas -dlcm=cg \
#	-Xptxas -dscm=wb -o $(TGT) $(SRC_C)
	nvcc $(NVCC_ARCH) -lineinfo -O3 -lib -rdc=true -Xptxas -dlcm=cg \
	-Xptxas -dscm=wb -Xptxas -maxrregcount=64 -o $(TGT) $(SRC_C)
#
#	nvcc $(ARCH) -O3 -lib -rdc=true -o $(TGT) $(SRC_C)

test:	$(TGT) makefile build-corr
	make -C tst/corr run-only

clean:
	rm -f $(TMP)
	make -C tst/common clean
	make -C tst/corr clean
	make -C tst/perf clean

build-perf:	$(TGT)
	make -C tst/common build
	make -C tst/perf build

build-corr:	$(TGT)
	make -C tst/common build
	make -C tst/corr build

build-test:	$(TGT)
	make -C tst/common build
	make -C tst/corr build
	make -C tst/perf build

install: $(HEADER) $(TGT)
	cp $(HEADER) $(PREFIX)/include/halloc.h
	cp $(TGT) $(PREFIX)/lib/libhalloc.a

uninstall:
	rm -f $(PREFIX)/include/halloc.h $(PREFIX)/lib/libhalloc.a
