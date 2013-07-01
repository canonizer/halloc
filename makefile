NAME=halloc
SRC_C=*.cu
SRC_H=
SRC=$(SRC_C) $(SRC_H)
TGT=$(NAME)
TMP=*~ \\\#* $(TGT)

build: $(TGT)
$(TGT):	$(SRC)
	nvcc -arch=sm_35 -O3 -Xcompiler -fopenmp $(SRC_C) -o $(TGT)

run: $(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
