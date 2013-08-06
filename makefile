NAME=halloc
SRC_C=src/*.cu
SRC_H=src/*.h src/*.cuh
SRC=$(SRC_C) $(SRC_H)
TGT=bin/$(NAME)
TMP=*~ \\\#* src/*~ src/\\\#* $(TGT)

build: $(TGT)
$(TGT):	$(SRC) makefile
#	nvcc -arch=sm_35 -O3 -g -G -rdc=true -Xcompiler -fopenmp $(SRC_C) -o $(TGT)
	nvcc -arch=sm_35 -O3 -rdc=true -Xcompiler -fopenmp $(SRC_C) -o $(TGT)

run: $(TGT)
	./$(TGT)

clean:
	rm -f $(TMP)
