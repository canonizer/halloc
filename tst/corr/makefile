TMP=*~

build:
	./make-all.sh build

clean:
	rm -f $(TMP)
	./make-all.sh clean

run: build
	./run-all-tests.pl

run-only:
	./run-all-tests.pl
