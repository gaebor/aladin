#ifndef INCLUDE_COMMON_H
#define INCLUDE_COMMON_H
#include <chrono>

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

#endif