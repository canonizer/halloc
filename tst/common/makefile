TGT=libcommontest.a
SRC_C=*.cu
SRC_H=*.h
SRC=$(SRC_C) $(SRC_H)
TMP=$(TGT) *~ \#*

build: $(TGT) 
$(TGT):	$(SRC) makefile
	nvcc -arch=sm_35 -O3 -lib -I../include -o $(TGT) $(SRC_C)

clean:
	rm -f $(TMP)
