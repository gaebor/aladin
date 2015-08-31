#ifndef LINEAR_STRUCTURES_H
#define LINEAR_STRUCTURES_H

namespace aladin
{
	template<class Number>
	struct MatrixHeader
	{
		MatrixHeader(Number* p, size_t r, size_t c, size_t r_s, size_t c_s=1)
			: ptr(p), col_step(c_s), row_step(r_s), cols(c), rows(r){}
		MatrixHeader(Number* p, size_t r, size_t c)
			: ptr(p), col_step(1), row_step(c), cols(c), rows(r){}
		Number* ptr;
		size_t col_step, row_step;
		size_t cols, rows;
		MatrixHeader t() const
		{
			MatrixHeader result(ptr, cols, rows, col_step, row_step);
			return result;
		}
		//! end ptr of matrix, assuming that the iteration is row-wise
		Number* end()const
		{
			return &((*this)(rows, 0));
		}
		Number& operator()(size_t i, size_t j)const
		{
			return ptr[i*row_step + j*col_step];
		}
		//! the pointer should be increased by this amount if one has been read a full row and wants to step to the beginning of the next row
		SIGNED_INT step_at_eor()const
		{
			return (SIGNED_INT)row_step - cols*col_step;
		}
		//! the pointer should be increased by this amount if one has been read a full column and wants to step to the top of the next column
		SIGNED_INT step_at_eoc()const
		{
			return (SIGNED_INT)col_step - rows*row_step;
		}
	};

	template <class Number>
	MatrixHeader<Number> make_header(Number* p, size_t r, size_t c)
	{
		return MatrixHeader<Number>(p, r, c, c, 1);
	}

	template <class Number>
	MatrixHeader<Number> make_header(Number* p, size_t r, size_t c, size_t r_s, size_t c_s)
	{
		return MatrixHeader<Number>(p, r, c, r_s, c_s);
	}

	template<class Number, template <class > class MyRing = RingProxy>
	class Matrix
	{
	public:
		typedef MyRing<Number> Ring;
		Matrix(size_t i, size_t j, const Number& value = Ring::AdditiveIdentity())
		:	elements(new Number[i*j]), isResponsible(true), rows(i), cols(j)
		{
			Number* elem = elements;
			for (size_t n = 0; n < rows*cols; ++n, ++elem)
			{
				*elem = value;
			}
		}
		Matrix(size_t i, size_t j, Number* value_ptr)
			:	elements(value_ptr), isResponsible(false), rows(i), cols(j)
		{
		}
		Matrix(const Matrix& other)
			:	elements(new Number[other.rows * other.cols]), isResponsible(true), rows(other.rows), cols(other.cols)
		{
			memcpy(elements, other.elements, sizeof(Number)*rows*cols);
		}
		Matrix& operator= (Matrix&& other)
		{
			if (isResponsible)
				delete[] elements;

			elements = other.elements;
			isResponsible = other.isResponsible;
			rows = other.rows;
			cols = other.cols;
			other.isResponsible = false;
		}
		~Matrix()
		{
			if (isResponsible)
				delete[] elements;
		}
		Number& operator()(size_t i, size_t j)
		{
			return elements[i*cols + j];
		}
	private:
		Number* elements;
		size_t rows;
		size_t cols;
		bool isResponsible;
	};
}

#endif