#ifndef ALADIN_TEMPLATE_META_LIBRARY_H
#define ALADIN_TEMPLATE_META_LIBRARY_H

namespace aladin
{

typedef size_t INT;

template<INT n, INT m>
struct max
{
public:
	static const INT result = n > m ? n : m;
};

template<INT n, INT m>
struct min
{
public:
	static const INT result = n < m ? n : m;
};

//! determines whether the first template argument is divisible with the second one.
template<INT n, INT m>
class Divisible
{
public:
	static const bool result = (n % m == 0);
};

//"template bind"
template<INT i>
class EvenNumber : public Divisible<i,2>
{
};


//!IF
template<bool condition, class Then, class Else>
struct IF
{
	typedef Else RET;
};

template<class Then, class Else>
struct IF<true,Then,Else>
{
	typedef Then RET;
};

template<class Class1, class Class2>
struct EQUAL
{
	static const bool result = false;
};

template<class Class1>
struct EQUAL<Class1, Class1>
{
	static const bool result = true;
};

//!class returns its template parameter
template<INT i>
struct identity
{
	static const INT result = i;
};

//!returns the first index in the range [i, end) for which Func returns true
template<INT i, INT end, template<INT> class Func>
class FindFirst
{
	static const bool AmIAHit = ((i >= end) || Func<i>::result);
public:
	static const INT result = IF<AmIAHit, identity<i>, FindFirst<i+1, end, Func>>::RET::result;
};

typedef IF<sizeof(void*)==sizeof(int), int, long long>::RET SIGNED_INT;

}//end of namespace aladin
#endif