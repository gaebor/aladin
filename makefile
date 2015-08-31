CC = gcc
CPP = g++
CFLAGS = -Iinc -lm -std=c++11 -O3 -march=native -Wall
SOURCE=src
apps = test test_cuda

all: $(apps)

test: src/test.cpp
	$(CPP) -o test $(CFLAGS) -lopenblas -I /cygdrive/e/PROGRAMOK/OpenBLAS/ $(SOURCE)/test.cpp -print-search-dirs

test_cuda: src/test.cpp
	$(CPP) -o test $(CFLAGS) -DUSE_CUDA $(SOURCE)/test.cpp

clean:
	rm -f ./test

doc:
	rm -R -f doxy/html
	doxygen doxy/config

#doxygen for windows
doc_w:
	rm -R -f doxy/html
	doxy/doxygen.exe doxy/config
	