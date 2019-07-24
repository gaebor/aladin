#ifndef ALADIN_H
#define ALADIN_H

/************************************************************************/
/* Abstract Linear Algebraic Derivations Including Numerics              */
/* borbely@math.bme.hu                                                  */
/************************************************************************/

#include <vector>
#include <memory>
#include <thread>
#include <algorithm>    // std::find
#include <functional>

#include "aladin/TemplateLibrary.h"
#include "aladin/linear_algebraic_structures.h"

#ifndef min
#	define min(x,y) (((x) < (y)) ? (x) : (y))
#	define MIN_HAS_BEEN_DEFINED_IN_ALADIN
#endif

namespace aladin{

//!computes A*B
template<bool add, class Number, class FMA>
bool gemm(const MatrixHeader<Number> mat1, const MatrixHeader<Number> mat2, const MatrixHeader<Number> result,
				FMA fma_f)
{
	if (mat1.cols != mat2.rows || result.rows != mat1.rows || result.cols != mat2.cols)
		return false;

	auto result_ptr = result.ptr;
	auto A_ptr = mat1.ptr;
	auto B_ptr = mat2.ptr;
	Number* result_dot_ptr;
	const Number* B_dot_ptr;

	size_t i,j,k;

    if (!add)
    {
        std::fill(result_ptr, result.ptr + result.rows*result.cols, Number(0));
    }

	for (i = 0; i < result.rows; ++i, result_ptr += result.row_step, B_ptr = mat2.ptr)
	{
        for (k = 0; k < mat1.cols; ++k, ++A_ptr, B_ptr += mat2.row_step)
        {
            result_dot_ptr = result_ptr;
            B_dot_ptr = B_ptr;
            for (j = 0; j < result.cols; ++j, ++result_dot_ptr, ++B_dot_ptr)
            {
                fma_f(*result_dot_ptr, *A_ptr, *B_dot_ptr);
            }
        }
		
	}
	return true;
}

//!computes A*B^T
template<bool add, class Number, class FMA>
bool gemm_t(const MatrixHeader<Number> mat1, const MatrixHeader<Number> mat2, const MatrixHeader<Number> result,
				 FMA fma_f)
{
	if (mat1.cols != mat2.cols || result.rows != mat1.rows || result.cols != mat2.rows)
		return false;

	auto result_ptr = result.ptr;
	auto A_ptr = mat1.ptr;
	auto B_ptr = mat2.ptr;
	auto A_dot_ptr = A_ptr;
	auto B_dot_ptr = B_ptr;
	const SIGNED_INT result_row_step = result.step_at_eor();

	size_t i,j,k;

	for (i = 0; i < result.rows; ++i, result_ptr += result_row_step, A_ptr += mat1.row_step, B_ptr = mat2.ptr)
	{
		for (j = 0; j < result.cols; ++j, result_ptr += result.col_step, B_ptr += mat2.row_step)
		{
			auto& thiselement = *result_ptr;
			if (!add)
				thiselement = 0;

			A_dot_ptr = A_ptr;
			B_dot_ptr = B_ptr;

			for (k = 0; k < mat1.cols; ++k, A_dot_ptr += mat1.col_step, B_dot_ptr += mat2.col_step)
			{
				// z += x*y
				fma_f(thiselement, *A_dot_ptr, *B_dot_ptr);
			}
		}
	}
	return true;
}

//!computes A*B, threadded
template<bool add, class Number, class FMA>
bool gemm_rowwise(const MatrixHeader<Number> mat1, const MatrixHeader<Number> mat2, const MatrixHeader<Number> result,
						FMA fma_f, size_t thread_number = 1)
{
	std::vector<std::shared_ptr<std::thread>> threads(thread_number, nullptr);
	std::vector<bool> thread_results(thread_number, false);

	int thread_id = 0;
	for (auto& thread_ptr : threads)
	{
		thread_ptr.reset(new std::thread([&](int id){
			size_t start_row = id * (mat1.rows/thread_number);
			size_t row_number = mat1.rows/thread_number;
			if (id == thread_number -1)
				row_number += mat1.rows % thread_number;

			auto a = aladin::make_header(&mat1(start_row, 0), row_number, mat1.cols, mat1.row_step, mat1.col_step);
			auto b = mat2;
			auto c = aladin::make_header(&result(start_row, 0), row_number, result.cols, result.row_step, result.col_step);
			thread_results[id] = aladin::gemm<add>(a, b, c, fma_f);
		}, thread_id
			));
		++thread_id;
	}
	for (auto thread_ptr : threads)
	{
		thread_ptr->join();
	}
	return std::find(thread_results.begin(), thread_results.end(), false) == thread_results.end();
}

template<bool add, class Number, class Subtract, class Mul, class Inverse>
size_t solve(const MatrixHeader<Number> mat,
	const Number AdditiveIdentity, const Number MultiplicativeIdentity,
	Mul Multiply, Inverse inv_f, Subtract sub_f)
{
	for (size_t i = 0; i < min(mat.rows, mat.cols); ++i)
	{
		auto& pivot = mat(i,i);

		if ( pivot == AdditiveIdentity)
		{
			return i;
			throw std::exception("Pivot is zero!");
		}

		const Number multiplier = inv_f(pivot);
		pivot = MultiplicativeIdentity;
		for (size_t k = i+1; k < mat.cols; ++k)
			mat(i, k) = Multiply(mat(i, k), multiplier);

		for (size_t j = 0; j < mat.rows; ++j)
		{
			auto& thisElement = mat(j, i);
			if (j == i)
				continue;
			
			for (size_t k = i+1; k < mat.cols; ++k)
				mat(j, k) = sub_f(mat(j, k), Multiply(thisElement, mat(i, k)));

			thisElement = AdditiveIdentity;
		}
	}
	return min(mat.rows, mat.cols);
}

////!Gaussian elimination
//template<class Number>
//static Number* inverse(Number* mat, const size_t rows)
//{
//	typedef FieldWrapper<Number> MyField;
//	typedef Position<Number,row_major> TakePosition;
//
//	Number* result = new Number[rows*rows];
//	for (size_t i = 0; i < rows; ++i)
//	{
//		for (size_t j = 0; j < rows; ++j)
//		{
//			if (i == j)
//				TakePosition::at(result, i,j, rows, rows) = MyField::MultiplicativeIdentity();
//			else
//				TakePosition::at(result, i,j, rows, rows) = MyField::AdditiveIdentity();
//		}
//	}
//
//	if (solve(mat, result, rows, rows, rows)==rows)
//		return result;
//	else
//	{
//		delete[] result;
//		return nullptr;
//	}
//}
//
////!Gaussian elimination
//template<class Number>
//static size_t solve(Number* mat1, Number* mat2, const size_t rows, const size_t cols1, const size_t cols2)
//{
//	typedef FieldWrapper<Number> MyField;
//	typedef Position<Number,row_major> TakePosition;
//
//	for (size_t i = 0; i < min(rows, cols1); ++i)
//	{
//		auto& pivot = TakePosition::at(mat1, i,i, rows, cols1);
//
//		if ( pivot == MyField::AdditiveIdentity())
//		{
//			return i;
//			throw std::exception("Pivot is zero!");
//		}
//
//		const auto multiplier = MyField::MultiplicativeInverse(pivot);
//		pivot = MyField::MultiplicativeIdentity();
//		for (size_t k = i+1; k < cols1; ++k)
//			TakePosition::at(mat1, i,k, rows, cols1) = MyField::Multiply(TakePosition::at(mat1, i,k, rows, cols1), multiplier);
//
//		for (size_t k = 0; k < cols2; ++k)
//			TakePosition::at(mat2, i,k, rows, cols2) = MyField::Multiply(TakePosition::at(mat2, i,k, rows, cols2), multiplier);
//
//		for (size_t j = 0; j < rows; ++j)
//		{
//			auto& thisElement = TakePosition::at(mat1, j,i, rows, cols1);
//			if (j == i)
//				continue;
//
//			for (size_t k = i+1; k < cols1; ++k)
//				TakePosition::at(mat1, j,k, rows, cols1) = MyField::Subtract(TakePosition::at(mat1, j,k, rows, cols1), MyField::Multiply(thisElement, TakePosition::at(mat1, i,k, rows, cols1)));
//
//			for (size_t k = 0; k < cols2; ++k)
//				TakePosition::at(mat2, j,k, rows, cols2) = MyField::Subtract(TakePosition::at(mat2, j,k, rows, cols2), MyField::Multiply(thisElement, TakePosition::at(mat2, i,k, rows, cols2)));
//
//			thisElement = MyField::AdditiveIdentity();
//		}
//	}
//
//	//todo find controversial equations
//	return min(rows, cols1);
//}
//
//};//end of eliminations

}//end of aladin

#ifdef MIN_HAS_BEEN_DEFINED_IN_ALADIN
	#undef min
#endif // MIN_HAS_BEEN_DEFINED_IN_ALADIN

#endif