# run vcvarsall.bat before nmake

# define path of precompiled dlls here
OPEN_BLAS_DIR=D:\OpenBLAS-v0.2.13-Win64-int64

BLAS_INC=$(OPEN_BLAS_DIR)\include
BLAS_LIB=$(OPEN_BLAS_DIR)\lib\libopenblas.dll.a

default: test
all: test doc

test: src\test.cpp
	cl src\test.cpp /O2 /Gy /Og /Ob2 /fp:fast /Ot /Iinc /I"$(BLAS_INC)" /link"$(BLAS_LIB)"

doc:
	RD /S /Q doxy\html
	doxy\doxygen.exe doxy\config
