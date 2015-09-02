CFLAGS = -Iinc -lm -lpthread -std=c++11 -O3 -march=native -Wall
CUDA_FLAGS = -Iinc -lm -std=c++11 -O3
apps = test test_cuda

all: $(apps)

test: src/test.cpp
	g++ -o test $(CFLAGS) -lopenblas src/test.cpp

test_cuda: src/test.cpp
	nvcc -o test_cuda $(CUDA_FLAGS) -DUSE_CUDA src/test.cpp -lcublas -lcuda

clean:
	rm -f ./test

doc:
	rm -R -f doxy/html
	doxygen doxy/config

#doxygen for windows
doc_w:
	rm -R -f doxy/html
	doxy/doxygen.exe doxy/config
	
