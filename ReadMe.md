# Abstract Linear Algebraic Derivations Including Numerics
contact author: `borbely@math.bme.hu`

This is a c++ template library to perform linear algebraic calculations with abstract objects.
featuring:
 * matrix multiplication over an arbitrary associative ring
 * Gaussian elimination of a matrix over an arbitrary filed
 * modulo prime calculations (Z_p)

and many more!

## Test
Use the test executables to measure performance.
Use the command line argument `--help` for clarification.

## build
### GCC

    make -f makefile

The [openblas](https://github.com/xianyi/OpenBLAS) library should be installed, example:

    sudo apt-get install libopenblas-dev

Set the environment variables to find the appropriate includes, example:

    CFLAGS=-I/opt/OpenBLAS/include make -f makefile

For the GPU test [cuda](https://developer.nvidia.com/cuda-downloads) library should be installed.

### Visual C++

    nmake /f make.mak

The [openblas](https://github.com/xianyi/OpenBLAS) library should be installed.
Set the `OPEN_BLAS_DIR` variable in `make.mak` to point to the compiled OpenBLAS home directory.

For the GPU test [cuda](https://developer.nvidia.com/cuda-downloads) library should be installed.
Also put the cuda installation library in the `PATH` during the compilation.
In runtime, you can put the appropriate `.dll`-s in your local directory.
