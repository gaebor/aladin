#ifndef INCLUDE_KERNELS_CUH
#define INCLUDE_KERNELS_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"

#include <cfloat> //ceil
#include <complex>

template<class Type>
void cuda_prod_imp(int blocks, const Sizes& sizes, const Type* A_dev, const Type* B_dev, Type* C_dev, bool transpose);

template<class Type>
void calculate_prod_reference(const Sizes& sizes, const Type* A, const Type* B, Type* C, bool transpose)
{
	static Type* pointers[] = { nullptr, nullptr, nullptr };
	static size_t allocatedSizes[] = { 0, 0, 0 };
	static size_t newSizes[] = { 0, 0, 0 };

	newSizes[0] = sizeof(Type)*sizes.row1*sizes.col1;
	newSizes[1] = sizeof(Type)*sizes.col1*sizes.col2;
	newSizes[2] = sizeof(Type)*sizes.row1*sizes.col2;

	for (int i = 0; i < 3; ++i)
	{
		if (newSizes[i] == 0)
			return;
		if (allocatedSizes[i] < newSizes[i])
		{
			if (cudaMalloc(pointers + i, newSizes[i]) == cudaSuccess)
				allocatedSizes[i] = newSizes[i];
			else
			{
				allocatedSizes[i] = 0;
				return;
			}				
		}
		if (cudaMemcpy(pointers[i], i == 0 ? A : (i == 1 ? B : C), newSizes[i], cudaMemcpyHostToDevice) != cudaSuccess)
			return;
		
	}
	const int block_number = ceil((float)sizes.row1*sizes.col2 / 256);
	cuda_prod_imp(block_number, sizes, pointers[0], pointers[1], pointers[2], transpose);

	for (int i = 0; i < 3; ++i)
	{
		cudaMemcpy((void*)(i == 0 ? A : (i == 1 ? B : C)), pointers[i], newSizes[i], cudaMemcpyDeviceToHost);
	}
}

#define DECLARATION_MACRO(type, type_name, transpose) \
void call_cuda_kernel_##type_name##_##transpose(int, const type *a, const type *b, type *c, int row_a, int col_a_row_b, int col_b)

#define DECLARATION_MACRO_BOTH(type, type_name) DECLARATION_MACRO(type, type_name, ); DECLARATION_MACRO(type, type_name, transpose)

DECLARATION_MACRO_BOTH(float, float);
DECLARATION_MACRO_BOTH(double, double);
DECLARATION_MACRO_BOTH(short, short);
DECLARATION_MACRO_BOTH(int, int);
DECLARATION_MACRO_BOTH(long long, longlong);
DECLARATION_MACRO_BOTH(std::complex<float>, complex);
DECLARATION_MACRO_BOTH(std::complex<double>, complexdouble);

#undef DECLARATION_MACRO_BOTH
#undef DECLARATION_MACRO

template<>
void cuda_prod_imp(int blocks, const Sizes& sizes, const float* A_dev, const float* B_dev, float* C_dev, bool transpose)
{
	if (transpose)
		call_cuda_kernel_float_transpose(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
	else
		call_cuda_kernel_float_(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
}

template<>
void cuda_prod_imp(int blocks, const Sizes& sizes, const double* A_dev, const double* B_dev, double* C_dev, bool transpose)
{
	if (transpose)
		call_cuda_kernel_double_transpose(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
	else
		call_cuda_kernel_double_(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
}

template<>
void cuda_prod_imp(int blocks, const Sizes& sizes, const int* A_dev, const int* B_dev, int* C_dev, bool transpose)
{
	if (transpose)
		call_cuda_kernel_int_transpose(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
	else
		call_cuda_kernel_int_(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
}

template<>
void cuda_prod_imp(int blocks, const Sizes& sizes, const long long* A_dev, const long long* B_dev, long long* C_dev, bool transpose)
{
	if (transpose)
		call_cuda_kernel_longlong_transpose(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
	else
		call_cuda_kernel_longlong_(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
}

template<>
void cuda_prod_imp(int blocks, const Sizes& sizes, const short* A_dev, const short* B_dev, short* C_dev, bool transpose)
{
	if (transpose)
		call_cuda_kernel_short_transpose(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
	else
		call_cuda_kernel_short_(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
}

template<>
void cuda_prod_imp(int blocks, const Sizes& sizes, const std::complex<float>* A_dev, const std::complex<float>* B_dev, std::complex<float>* C_dev, bool transpose)
{
	if (transpose)
		call_cuda_kernel_complex_transpose(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
	else
		call_cuda_kernel_complex_(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
}

template<>
void cuda_prod_imp(int blocks, const Sizes& sizes, const std::complex<double>* A_dev, const std::complex<double>* B_dev, std::complex<double>* C_dev, bool transpose)
{
	if (transpose)
		call_cuda_kernel_complexdouble_transpose(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
	else
		call_cuda_kernel_complexdouble_(blocks, A_dev, B_dev, C_dev, sizes.row1, sizes.col1, sizes.col2);
}

#endif
