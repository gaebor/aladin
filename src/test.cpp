#include <iostream>
#include <random>
#include <complex>
#include <chrono>

#include <string.h>

#include "aladin/aladin.h"


struct Sizes
{
	int row1, col1, col2;
};

class Timer
{
private:
	typedef std::chrono::high_resolution_clock my_clock;
	typedef std::chrono::time_point<my_clock> time_point;
	typedef std::chrono::duration<double> duration_t;
public:
	Timer()
	{
		tick();
	}
	void tick()
	{
		start = my_clock::now();
	}
	double tack()const
	{
		duration_t elapsed_seconds = my_clock::now() - start;
		return elapsed_seconds.count();
	}

private:
	time_point start;
};

#ifndef USE_CUDA
#	include "cblas.h"

template<class Type>
void calculate_prod_reference(const Sizes& sizes, const Type* A, const Type* B, Type* C2, bool transpose)
{
}

template<>
void calculate_prod_reference<double>(const Sizes& sizes, const double* A, const double* B, double* C2, bool transpose)
{
	if (!transpose)
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			sizes.row1, sizes.col2, sizes.col1, 1.0, A, sizes.col1, B, sizes.col2, 0.0, C2, sizes.col2);
	else
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			sizes.row1, sizes.col2, sizes.col1, 1.0, A, sizes.col1, B, sizes.col1, 0.0, C2, sizes.col2);
}

template<>
void calculate_prod_reference<float>(const Sizes& sizes, const float* A, const float* B, float* C2, bool transpose)
{
	if (!transpose)
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			sizes.row1, sizes.col2, sizes.col1, 1.0, A, sizes.col1, B, sizes.col2, 0.0, C2, sizes.col2);
	else
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			sizes.row1, sizes.col2, sizes.col1, 1.0, A, sizes.col1, B, sizes.col1, 0.0, C2, sizes.col2);
}

template<>
void calculate_prod_reference<std::complex<float>>(const Sizes& sizes, const std::complex<float>* A, const std::complex<float>* B, std::complex<float>* C2, bool transpose)
{
	static const float alpha[] = { 1.0f , 0.0f};
	static const float beta[] = { 0.0f, 0.0f };
	if (!transpose)
		cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			sizes.row1, sizes.col2, sizes.col1, alpha,
			reinterpret_cast<const float*>(A), sizes.col1,
			reinterpret_cast<const float*>(B), sizes.col2,
			beta, reinterpret_cast<float*>(C2), sizes.col2);
	else
		cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			sizes.row1, sizes.col2, sizes.col1, alpha,
			reinterpret_cast<const float*>(A), sizes.col1,
			reinterpret_cast<const float*>(B), sizes.col1,
			beta, reinterpret_cast<float*>(C2), sizes.col2);
}

template<>
void calculate_prod_reference<std::complex<double>>(const Sizes& sizes, const std::complex<double>* A, const std::complex<double>* B, std::complex<double>* C2, bool transpose)
{
	static const double alpha[] = { 1.0, 0.0 };
	static const double beta[] = { 0.0, 0.0 };
	if (!transpose)
		cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			sizes.row1, sizes.col2, sizes.col1, alpha,
			reinterpret_cast<const double*>(A), sizes.col1,
			reinterpret_cast<const double*>(B), sizes.col2,
			beta, reinterpret_cast<double*>(C2), sizes.col2);
	else
		cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
			sizes.row1, sizes.col2, sizes.col1, alpha,
			reinterpret_cast<const double*>(A), sizes.col1,
			reinterpret_cast<const double*>(B), sizes.col1,
			beta, reinterpret_cast<double*>(C2), sizes.col2);
}

#else
#	include "cuda_runtime.h"
#	include "device_launch_parameters.h"
#	include "cublas_v2.h"

template<class Type>
void cuda_prod_imp(cublasHandle_t handle, const Sizes& sizes, const Type* A_dev, const Type* B_dev, Type* C_dev, bool transpose)
{
}

template<class Type>
void calculate_prod_reference(const Sizes& sizes, Type* A, Type* B, Type* C, bool transpose)
{
	static cudaError_t error;
	static cublasStatus_t stat;
	static cublasHandle_t handle;
	static Type* pointers[] = { nullptr, nullptr, nullptr };
	static size_t allocatedSizes[] = { 0, 0, 0 };
	static size_t newSizes[] = { 0, 0, 0 };
	static bool CUDA_OK = (stat = cublasCreate (& handle )) == CUBLAS_STATUS_SUCCESS;

	if (!CUDA_OK)
		return;
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

	}
	stat = cublasSetMatrix (sizes.col1, sizes.row1, sizeof(Type), A, sizes.col1, pointers[0], sizes.col1);
	stat = cublasSetMatrix (sizes.col2, sizes.col1, sizeof(Type), B, sizes.col2, pointers[1], sizes.col2);
	stat = cublasSetMatrix (sizes.col2, sizes.row1, sizeof(Type), C, sizes.col2, pointers[2], sizes.col2);

	cuda_prod_imp(handle, sizes, pointers[0], pointers[1], pointers[2], transpose);

	stat = cublasGetMatrix (sizes.col1, sizes.row1, sizeof(Type), pointers[0], sizes.col1, A, sizes.col1);
	stat = cublasGetMatrix (sizes.col2, sizes.col1, sizeof(Type), pointers[1], sizes.col2, B, sizes.col2);
	stat = cublasGetMatrix (sizes.col2, sizes.row1, sizeof(Type), pointers[2], sizes.col2, C, sizes.col2);
}

template<>
void cuda_prod_imp(cublasHandle_t handle, const Sizes& sizes, const float* A_dev, const float* B_dev, float* C_dev, bool transpose)
{
	static cudaError_t error;
	static cublasStatus_t stat;
	static const float beta = 0.0f;
	static const float alpha = 1.0f;

	/************************************************************************/
	/* HASK ALERT! This is something that you don't want to know            */
	/************************************************************************/
	if (transpose)
		stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, sizes.col2, sizes.row1, sizes.col1, &alpha, B_dev, sizes.col1, A_dev, sizes.col1, &beta, C_dev, sizes.col2);
	else
		stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sizes.col2, sizes.row1, sizes.col1, &alpha, B_dev, sizes.col2, A_dev, sizes.col1, &beta, C_dev, sizes.col2);
}

template<>
void cuda_prod_imp(cublasHandle_t handle, const Sizes& sizes, const double* A_dev, const double* B_dev, double* C_dev, bool transpose)
{
        static cudaError_t error;
        static cublasStatus_t stat;
        static const double beta = 0.0;
        static const double alpha = 1.0;

        if (transpose)
			stat = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, sizes.col2, sizes.row1, sizes.col1, &alpha, B_dev, sizes.col1, A_dev, sizes.col1, &beta, C_dev, sizes.col2);
        else
			stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sizes.col2, sizes.row1, sizes.col1, &alpha, B_dev, sizes.col2, A_dev, sizes.col1, &beta, C_dev, sizes.col2);
}

template<>
void cuda_prod_imp(cublasHandle_t handle, const Sizes& sizes, const std::complex<float>* A_dev, const std::complex<float>* B_dev, std::complex<float>* C_dev, bool transpose)
{
	static cudaError_t error;
	static cublasStatus_t stat;
	static const cuComplex beta = { 0.0f, 0.0f };
	static const cuComplex alpha = { 1.0f , 0.0f };

	if (transpose)
		stat = cublasCgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, sizes.col2, sizes.row1, sizes.col1, &alpha,
		reinterpret_cast<const cuComplex*>(B_dev), sizes.col1,
		reinterpret_cast<const cuComplex*>(A_dev), sizes.col1,
		&beta, reinterpret_cast<cuComplex*>(C_dev), sizes.col2);
	else
		stat = cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sizes.col2, sizes.row1, sizes.col1, &alpha,
		reinterpret_cast<const cuComplex*>(B_dev), sizes.col2,
		reinterpret_cast<const cuComplex*>(A_dev), sizes.col1, &beta,
		reinterpret_cast<cuComplex*>(C_dev), sizes.col2);
}

template<>
void cuda_prod_imp(cublasHandle_t handle, const Sizes& sizes, const std::complex<double>* A_dev, const std::complex<double>* B_dev, std::complex<double>* C_dev, bool transpose)
{
	static cudaError_t error;
	static cublasStatus_t stat;
	static const cuDoubleComplex beta = { 0.0, 0.0 };
	static const cuDoubleComplex alpha = { 1.0, 0.0 };

	if (transpose)
		stat = cublasZgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, sizes.col2, sizes.row1, sizes.col1, &alpha,
		reinterpret_cast<const cuDoubleComplex*>(B_dev), sizes.col1,
		reinterpret_cast<const cuDoubleComplex*>(A_dev), sizes.col1,
		&beta, reinterpret_cast<cuDoubleComplex*>(C_dev), sizes.col2);
	else
		stat = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, sizes.col2, sizes.row1, sizes.col1, &alpha,
		reinterpret_cast<const cuDoubleComplex*>(B_dev), sizes.col2,
		reinterpret_cast<const cuDoubleComplex*>(A_dev), sizes.col1, &beta,
		reinterpret_cast<cuDoubleComplex*>(C_dev), sizes.col2);
}

#endif

template<class Type>
double check_reference(const Sizes& sizes, const Type* X1, const Type* X2)
{
	double error = 0;
	for (size_t i = 0; i < sizes.row1*sizes.col2; ++i)
	{
		error += std::pow(std::abs(X1[i]-X2[i]),2);
	}
	return std::sqrt(error/(sizes.row1*sizes.col2)); 
}

template<class Container>
typename Container::value_type Mean(const Container& v)
{
	typename Container::value_type total=0;
	for (const auto& value : v)
	{
		total += value;
	}
	return total/v.size();
}

struct TestEntry
{
	Sizes sizes;
	size_t epochs;
	int threads;
	bool transpose;
	double aladin_time;
	double blas_time;
	double error;
};

template<class Type>
void init_matrices(Type* A, Type*B, const Sizes& sizes);

template<class Type>
void init_matrices_float(Type* A, Type* B, const Sizes& sizes)
{
	static std::random_device rd;
	static std::default_random_engine generator(rd());
	static std::uniform_real_distribution<Type> distribution(-1.0, 1.0);
	for (int i = 0; i < sizes.row1*sizes.col1; ++i)
	{
		A[i] = distribution(generator);
	}
	for (int i = 0; i < sizes.col1*sizes.col2; ++i)
	{
		B[i] = distribution(generator);
	}
}

template<class Type>
void init_matrices(Type* A, Type*B, const Sizes& sizes)
{
	static std::random_device rd;
	static std::default_random_engine generator(rd());
	static std::uniform_int_distribution<Type> distribution(-10, 10);
	for (int i = 0; i < sizes.row1*sizes.col1; ++i)
	{
		A[i] = distribution(generator);
	}
	for (int i = 0; i < sizes.col1*sizes.col2; ++i)
	{
		B[i] = distribution(generator);
	}
}

template<>
void init_matrices<std::complex<float>>(std::complex<float>* A, std::complex<float>*B, const Sizes& sizes)
{
	static std::random_device rd;
	static std::default_random_engine generator(rd());
	static std::uniform_real_distribution<float> distribution(-1.0f,1.0f);

	for (int i = 0; i < sizes.row1*sizes.col1; ++i)
	{
		A[i].real(distribution(generator));
		A[i].imag(distribution(generator));
	}

	for (int i = 0; i < sizes.col1*sizes.col2; ++i)
	{
		B[i].real(distribution(generator));
		B[i].imag(distribution(generator));
	}
}

template<>
void init_matrices<std::complex<double>>(std::complex<double>* A, std::complex<double>*B, const Sizes& sizes)
{
	static std::random_device rd;
	static std::default_random_engine generator(rd());
	static std::uniform_real_distribution<double> distribution(-1.0, 1.0);

	for (int i = 0; i < sizes.row1*sizes.col1; ++i)
	{
		A[i].real(distribution(generator));
		A[i].imag(distribution(generator));
	}

	for (int i = 0; i < sizes.col1*sizes.col2; ++i)
	{
		B[i].real(distribution(generator));
		B[i].imag(distribution(generator));
	}
}

template<>
void init_matrices<float>(float* A, float*B, const Sizes& sizes)
{
	static std::random_device rd;
	static std::default_random_engine generator(rd());
	static std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

	for (int i = 0; i < sizes.row1*sizes.col1; ++i)
	{
		A[i] = distribution(generator);
	}
	for (int i = 0; i < sizes.col1*sizes.col2; ++i)
	{
		B[i] = distribution(generator);
	}
}

template<>
void init_matrices<double>(double* A, double*B, const Sizes& sizes)
{
	static std::random_device rd;
	static std::default_random_engine generator(rd());
	static std::uniform_real_distribution<double> distribution(-1.0, 1.0);

	for (int i = 0; i < sizes.row1*sizes.col1; ++i)
	{
		A[i] = distribution(generator);
	}
	for (int i = 0; i < sizes.col1*sizes.col2; ++i)
	{
		B[i] = distribution(generator);
	}
}

template<class Type>
void prod_test(TestEntry& result)
{
	Timer timer;
	auto sizes = result.sizes;

	std::vector<double> calculation_times;
	std::vector<double> baseline_times;

	std::vector<double> checking_errors;

	Type *A, *B, *C1, *C2;

#ifndef USE_CUDA
	openblas_set_num_threads(result.threads);

	A = new Type[sizes.row1*sizes.col1];
	B = new Type[sizes.col1*sizes.col2];
	C1 = new Type[sizes.row1*sizes.col2];
	C2 = new Type[sizes.row1*sizes.col2];

	if (A==nullptr || B == nullptr || C1 == nullptr || C2 == nullptr)
	{
		std::cerr << "unable to allocate memory" << std::endl;
		return;
	}
	std::cout << "aladin time\tblas time\terror" << std::endl;
#else

	cudaError_t cuda_error;
	cuda_error = cudaHostAlloc(&A, sizeof(Type)*sizes.row1*sizes.col1, cudaHostAllocDefault);
	if (cuda_error != cudaSuccess)
	{
		std::cerr << "Unable to allocate pinned memory" << std::endl;
		return;
	}
	cuda_error = cudaHostAlloc(&B, sizeof(Type)*sizes.col1*sizes.col2, cudaHostAllocDefault);
	if (cuda_error != cudaSuccess)
	{
		std::cerr << "Unable to allocate pinned memory" << std::endl;
		return;
	}
	cuda_error = cudaHostAlloc(&C1, sizeof(Type)*sizes.row1*sizes.col2, cudaHostAllocDefault);
	if (cuda_error != cudaSuccess)
	{
		std::cerr << "Unable to allocate pinned memory" << std::endl;
		return;
	}
	cuda_error = cudaHostAlloc(&C2, sizeof(Type)*sizes.row1*sizes.col2, cudaHostAllocDefault);
	if (cuda_error != cudaSuccess)
	{
		std::cerr << "Unable to allocate pinned memory" << std::endl;
		return;
	}
	std::cout << "aladin time\tgpu time\terror" << std::endl;
#endif

	for (size_t e = 0; e < result.epochs; ++e)
	{
		init_matrices(A, B, sizes);

		timer.tick();
		if (result.transpose)
		{
			aladin::gemm_t<false>(aladin::make_header(A, sizes.row1, sizes.col1), aladin::make_header(B, sizes.col2, sizes.col1), aladin::make_header(C1, sizes.row1, sizes.col2),
				[](Type& one, const Type& other){one += other;}, [](const Type& one, const Type& other){return one*other;});
		}else{
			aladin::gemm_rowwise<false>(aladin::make_header(A, sizes.row1, sizes.col1), aladin::make_header(B, sizes.col1, sizes.col2), aladin::make_header(C1, sizes.row1, sizes.col2),
				[](Type& one, const Type& other){one += other;}, [](const Type& one, const Type& other){return one*other;},
				result.threads);
		}
		calculation_times.push_back(timer.tack());

		timer.tick();
		calculate_prod_reference(sizes, A, B, C2, result.transpose);
		baseline_times.push_back(timer.tack());

		checking_errors.push_back(check_reference(sizes, C1, C2));

		std::cerr << '\r' << Mean(calculation_times) << '\t' << Mean(baseline_times) << '\t' << Mean(checking_errors) << '\t';
	}
	std::cerr << std::endl;

	result.aladin_time = Mean(calculation_times);
	result.blas_time = Mean(baseline_times);
	result.error= Mean(checking_errors);

#ifdef USE_CUDA
	cudaFreeHost(A);
	cudaFreeHost(B);
	cudaFreeHost(C1);
	cudaFreeHost(C2);
#else
	delete[] A;
	delete[] B;
	delete[] C1;
	delete[] C2;
#endif // USE_CUDA

}

int main(int argc, char* argv[])
{
	TestEntry test;
	test.sizes.row1 = 100;
	test.sizes.col1 = 200;
	test.sizes.col2 = 300;
	test.transpose = false;

	test.threads = 1;
	test.epochs = 10;
	std::string type = "double";
	
	++argv; //name of executable

#ifdef USE_CUDA
	if (cudaSetDevice(0) != cudaSuccess)
	{
		std::cerr << "unable to select graphic card!" << std::endl;
		return -1;
	}
#endif

	for( ; *argv != NULL; ++argv)
	{
		if (strcmp(*argv, "--help") == 0)
		{
			std::cout << "this is a performance test app, usage: test [options]" << std::endl;
			std::cout << "options:" << std::endl;
			std::cout << "\t-s int [int int]\tsize of matrices.\n\t\tIf one positive number follows this option, then square matrices are obtained." <<
				"\n\t\tIf 3 positive numbers follow like \"-s n k m\" then A=n*k, B=k*m, A*B=n*m." << 
				"\n\t\tDefault is: -s " << test.sizes.row1 << " " << test.sizes.col1 << " " << test.sizes.col2  << std::endl;
			std::cout << "\t-t int\tthreads, default is: " << test.threads << std::endl;
			std::cout << "\t-e int\tepoch, number of identical tests, default is: " << test.epochs<< std::endl;
			std::cout << "\t-h\ttranspose, A*B' is calculated, default is " << (test.transpose ? "enabled" : "disabled") << std::endl;
			std::cout << "\t\trest of the parameters specify the type, it can be:\n";
			for (auto s : { "float", "double", "complex", "complexdouble", "short", "int", "longlong" })
			{
				std::cout << "\t\t" << s << "\n";
			}
			std::cout << "\t\tdefault type is \"" << type <<"\"" << std::endl;
			return 1;
		}
		else if (strcmp(*argv, "-s") == 0)
		{
			int n = 0; //number of integers after -s flag until any other flag
			std::vector<int> s(argc);
			while (argv[1] != NULL && (s[n] = atoi(argv[1])) > 0)
			{
				++n;
				++argv;
			}
			if (n == 1)
			{
				test.sizes.row1 = test.sizes.col1 = test.sizes.col2 = s[0];
			}else if (n ==3)
			{
				test.sizes.row1 = s[0];
				test.sizes.col1 = s[1];
				test.sizes.col2 = s[2];
			}
		}else if (strcmp(*argv, "-t") == 0)
		{
			test.threads = atoi(*++argv);
		}else if (strcmp(*argv, "-e") == 0){
			test.epochs = atoi(*++argv);
		}else if (strcmp(*argv, "-h") == 0){
			test.transpose = true;
		}else
			type = *argv;
	}
	
	if (type == "float")
	{
		prod_test<float>(test);
	}else if (type == "double")
	{
		prod_test<double>(test);
	}
	else if (type == "short")
	{
		prod_test<short>(test);
	}
	else if (type == "int")
	{
		prod_test<int>(test);
	}
	else if (type == "longlong")
	{
		prod_test<long long>(test);
	}
	else if (type == "complex")
	{
		prod_test<std::complex<float>>(test);
	}
	else if (type == "complexdouble")
	{
		prod_test<std::complex<double>>(test);
	}

	std::cout << test.aladin_time << '\t' << test.blas_time << '\t' << test.error << std::endl;

	return 0;
}
