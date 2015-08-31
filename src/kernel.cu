#include "kernels.cuh"

#define DEFINE_KERNEL_MACRO_TRANSPOSE(type, name)\
__global__ void dot_kernel_##name(const type *a, const type *b, type *c, int row_a, int col_a_row_b, int col_b) \
{                                                      \
	int idx = blockIdx.x * blockDim.x + threadIdx.x;   \
	int i;                                             \
	const int target_row = idx / col_b;                \
	const int target_col = idx % col_b;                \
	if (idx >= row_a*col_b)                            \
		return;                                        \
	c += idx;                                          \
	*c = 0;                                            \
	a += target_row*col_a_row_b;                       \
	b += target_col*col_b;                             \
	for (i = 0; i < col_a_row_b; ++i, a += 1, b += 1)  \
		*c += *a + *b;                                 \
}

#define DEFINE_KERNEL_MACRO(type, name)\
__global__ void dot_kernel_##name(const type *a, const type *b, type *c, int row_a, int col_a_row_b, int col_b) \
{                                                      \
	int idx = blockIdx.x * blockDim.x + threadIdx.x;   \
	int i;                                             \
	const int target_row = idx / col_b;                \
	const int target_col = idx % col_b;                \
	if (idx >= row_a*col_b)                            \
		return;                                        \
	c += idx;                                          \
	*c = 0;                                            \
	a += target_row*col_a_row_b;                       \
	b += target_col;                                   \
	for (i = 0; i < col_a_row_b; ++i, a+=1, b+=col_b)  \
		*c += *a + *b;                                 \
}

#define CALL_CUDA_MACRO(type, name) \
DEFINE_KERNEL_MACRO(type, name##_) \
DEFINE_KERNEL_MACRO_TRANSPOSE(type, name##_transpose) \
void call_cuda_kernel_##name##_(int blocks, const type *a, const type *b, type *c, int row_a, int col_a_row_b, int col_b) \
{dot_kernel_##name##_<<<blocks, 256>> >(a, b, c, row_a, col_a_row_b, col_b);}\
void call_cuda_kernel_##name##_transpose(int blocks, const type *a, const type *b, type *c, int row_a, int col_a_row_b, int col_b) \
{dot_kernel_##name##_transpose<<<blocks, 256>> >(a, b, c, row_a, col_a_row_b, col_b);}

CALL_CUDA_MACRO(float, float)
CALL_CUDA_MACRO(double, double)
CALL_CUDA_MACRO(long long, longlong)
CALL_CUDA_MACRO(short, short)
CALL_CUDA_MACRO(int, int)

#define DEFINE_KERNEL_MACRO_COMPLEX_TRANSPOSE(type, name)\
__global__ void dot_kernel_##name(const type *a, const type *b, type *c, int row_a, int col_a_row_b, int col_b) \
{                                                      \
	int idx = blockIdx.x * blockDim.x + threadIdx.x;   \
	int i;                                             \
	const int target_row = idx / col_b;                \
	const int target_col = idx % col_b;                \
	if (idx >= row_a*col_b)                            \
		return;                                        \
	c += idx*2;                                        \
	c[0] = 0;                                          \
	c[1] = 0;                                          \
	a += target_row*col_a_row_b*2;                     \
	b += target_col*col_b*2;                           \
	for (i = 0; i < col_a_row_b; ++i, a += 2, b += 2)  \
	{                                                  \
		c[0] += a[0] * b[0] - a[1] * b[1];             \
		c[1] += a[1] * b[0] + a[0] * b[1];             \
	}                                                  \
}

#define DEFINE_KERNEL_MACRO_COMPLEX(type, name)\
__global__ void dot_kernel_##name(const type *a, const type *b, type *c, int row_a, int col_a_row_b, int col_b) \
{                                                      \
	int idx = blockIdx.x * blockDim.x + threadIdx.x;   \
	int i;                                             \
	const int target_row = idx / col_b;                \
	const int target_col = idx % col_b;                \
	if (idx >= row_a*col_b)                            \
		return;                                        \
	c += idx*2;                                        \
	c[0] = 0;                                          \
	c[1] = 0;                                          \
	a += target_row*col_a_row_b*2;                     \
	b += target_col*2;                                 \
	for (i = 0; i < col_a_row_b; ++i, a+=2, b+=col_b*2)\
	{                                                  \
		c[0] += a[0] * b[0] - a[1] * b[1];             \
		c[1] += a[1] * b[0] + a[0] * b[1];             \
	}                                                  \
}

#define CALL_CUDA_MACRO_COMPLEX(type, name) \
DEFINE_KERNEL_MACRO_COMPLEX(type, name##_) \
DEFINE_KERNEL_MACRO_COMPLEX_TRANSPOSE(type, name##_transpose) \
void call_cuda_kernel_##name##_(int blocks, const std::complex<type> *a, const std::complex<type> *b, std::complex<type> *c, int row_a, int col_a_row_b, int col_b) \
{dot_kernel_##name##_<<<blocks, 256>> >(reinterpret_cast<const type*>(a), reinterpret_cast<const type*>(b), reinterpret_cast<type*>(c), row_a, col_a_row_b, col_b);}\
void call_cuda_kernel_##name##_transpose(int blocks, const std::complex<type> *a, const std::complex<type> *b, std::complex<type> *c, int row_a, int col_a_row_b, int col_b) \
{dot_kernel_##name##_transpose<<<blocks, 256>> >(reinterpret_cast<const type*>(a), reinterpret_cast<const type*>(b), reinterpret_cast<type*>(c), row_a, col_a_row_b, col_b);}

CALL_CUDA_MACRO_COMPLEX(float, complex)
CALL_CUDA_MACRO_COMPLEX(double, complexdouble)
