CFLAGS+= -Iinc -pthread -std=c++11 -O3 -march=native -Wall
apps = test test_cuda

all: $(apps)

test: src/test.cpp
	g++ -o test $(CFLAGS) src/test.cpp -lopenblas

test_cuda: src/test.cpp
	nvcc -o test_cuda $(CFLAGS) -DUSE_CUDA src/test.cpp -lcublas -lcuda

clean:
	rm -f $(apps)

doc:
	rm -R -f doxy/html
	doxygen doxy/config
