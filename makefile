CFLAGS = -Iinc -lm -std=c++11 -O3 -march=native -Wall
CUDA_FLAGS = -Iinc -lm -std=c++11 -O3
apps = test test_cuda

all: $(apps)

test: src/test.cpp
	$(CPP) -o test $(CFLAGS) -lopenblas -I /cygdrive/e/PROGRAMOK/OpenBLAS/ src/test.cpp

test_cuda: src/test.cpp src/kernel.cu
	nvcc -o test_cuda $(CUDA_FLAGS) -DUSE_CUDA src/test.cpp src/kernel.cu

clean:
	rm -f ./test

doc:
	rm -R -f doxy/html
	doxygen doxy/config

#doxygen for windows
doc_w:
	rm -R -f doxy/html
	doxy/doxygen.exe doxy/config
	