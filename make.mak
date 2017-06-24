# run vcvarsall.bat before nmake

# define path of precompiled dlls here
OPEN_BLAS_DIR=D:\OpenBLAS-v0.2.14-Win64-int64

BLAS_INC=$(OPEN_BLAS_DIR)\include
BLAS_LIB=$(OPEN_BLAS_DIR)\lib\libopenblas.dll.a
apps=test test_cuda

CFLAGS=/Ox /fp:fast /Ot /Iinc /EHsc
CUDA_FLAGS=-Iinc -O3

default: test
all: test doc

test: src\test.cpp
	cl src\test.cpp $(CFLAGS) /I"$(BLAS_INC)" /link"$(BLAS_LIB)"
    copy /Y $(OPEN_BLAS_DIR)\bin\libopenblas.dll .

test_cuda: src\test.cpp
	nvcc -o test_cuda $(CUDA_FLAGS) -DUSE_CUDA src/test.cpp -lcublas -lcuda

doc:
	RD /S /Q doxy\html
	doxy\doxygen.exe doxy\config

clean:
	del /Q *.exe *.lib *.exp *.obj
