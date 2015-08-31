
//#define _USE_MATH_DEFINES
//
//#include <math.h>
//#include <complex>

#include <iostream>
#include <chrono>
#include <random>
#include <complex>

#ifndef USE_CUDA
#	include "cblas.h"
#else
#	include "kernels.cuh"
#endif

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

template<class Type>
void calculate_prod_reference(const Sizes& sizes, const Type* A, const Type* B, Type* C2, bool transpose);

template<>
void calculate_prod_reference<double>(const Sizes& sizes, const double* A, const double* B, double* C2, bool transpose)
{

	cblas_dgemm(CblasRowMajor, CblasNoTrans, transpose ? CblasTrans : CblasNoTrans,
		sizes.row1, sizes.col2, sizes.col1, 1.0, A, sizes.col1, B, sizes.col2, 0.0, C2, sizes.col2);
}

template<>
void calculate_prod_reference<float>(const Sizes& sizes, const float* A, const float* B, float* C2, bool transpose)
{
	cblas_sgemm(CblasRowMajor, CblasNoTrans, transpose ? CblasTrans : CblasNoTrans,
		sizes.row1, sizes.col2, sizes.col1, 1.0, A, sizes.col1, B, sizes.col2, 0.0, C2, sizes.col2);
}

template<>
void calculate_prod_reference<std::complex<float>>(const Sizes& sizes, const std::complex<float>* A, const std::complex<float>* B, std::complex<float>* C2, bool transpose)
{
	float alpha = 1.0;
	float beta = 0.0;
	cblas_cgemm(CblasRowMajor, CblasNoTrans, transpose ? CblasTrans : CblasNoTrans,
		sizes.row1, sizes.col2, sizes.col1, &alpha, A->_Val, sizes.col1, B->_Val, sizes.col2, &beta, C2->_Val, sizes.col2);
}

template<>
void calculate_prod_reference<std::complex<double>>(const Sizes& sizes, const std::complex<double>* A, const std::complex<double>* B, std::complex<double>* C2, bool transpose)
{
	double alpha = 1.0;
	double beta = 0.0;
	cblas_zgemm(CblasRowMajor, CblasNoTrans, transpose ? CblasTrans : CblasNoTrans,
		sizes.row1, sizes.col2, sizes.col1, &alpha, A->_Val, sizes.col1, B->_Val, sizes.col2, &beta, C2->_Val, sizes.col2);
}

template<>
void calculate_prod_reference<short>(const Sizes& sizes, const short* A, const short* B, short* C2, bool transpose)
{
}

template<>
void calculate_prod_reference<int>(const Sizes& sizes, const int* A, const int* B, int* C2, bool transpose)
{
}

template<>
void calculate_prod_reference<long long>(const Sizes& sizes, const long long* A, const long long* B, long long* C2, bool transpose)
{
}

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
void init_matrices(Type* A, Type*B, const Sizes& sizes)
{
	static std::random_device rd;
	static std::default_random_engine generator(rd());
	static std::uniform_real_distribution<Type> distribution(-10,10);

	for (int i = 0; i < sizes.row1*sizes.col1; ++i)
		A[i] = distribution(generator);

	for (int i = 0; i < sizes.col1*sizes.col2; ++i)
		B[i] = distribution(generator);
}

template<>
void init_matrices<std::complex<float>>(std::complex<float>* A, std::complex<float>*B, const Sizes& sizes)
{
	static std::random_device rd;
	static std::default_random_engine generator(rd());
	static std::uniform_real_distribution<float> distribution(-10,10);

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
	static std::uniform_real_distribution<double> distribution(-10,10);

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
void init_matrices<short>(short* A, short*B, const Sizes& sizes)
{
	static std::random_device rd;
	static std::default_random_engine generator(rd());
	static std::uniform_int_distribution<short> distribution(-10,10);

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
void init_matrices<int>(int* A, int*B, const Sizes& sizes)
{
	static std::random_device rd;
	static std::default_random_engine generator(rd());
	static std::uniform_int_distribution<int> distribution(-10,10);

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
void init_matrices<long long>(long long* A, long long*B, const Sizes& sizes)
{
	static std::random_device rd;
	static std::default_random_engine generator(rd());
	static std::uniform_int_distribution<long long> distribution(-10,10);

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

#ifndef USE_CUDA
	openblas_set_num_threads(result.threads);
#else

#endif

	auto A = new Type[sizes.row1*sizes.col1];
	auto B = new Type[sizes.col1*sizes.col2];
	auto C1 = new Type[sizes.row1*sizes.col2];
	auto C2 = new Type[sizes.row1*sizes.col2];

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

	delete[] A;
	delete[] B;
	delete[] C1;
	delete[] C2;
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

	for( ; *argv != NULL; ++argv)
	{
		if (strcmp(*argv, "-s") == 0)
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
