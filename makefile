NAME=halloc
SRC_C=*.cu
SRC_H=*.h
SRC=$(SRC_C) $(SRC_H)
TGT=$(NAME)
TMP=*~ \\\#* $(TGT)

build: $(TGT)
$(TGT):	$(SRC) makefile
	nvcc -arch=sm_35 -O3 -rdc=true -Xcompiler -fopenmp $(SRC_C) -o $(TGT)

run: $(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
