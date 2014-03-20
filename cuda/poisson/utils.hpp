
#ifndef __JACB_UTILS_H
#define __JACB_UTILS_H

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <fstream>
#include <cstddef>

struct IterationStats
{

	float error;
	int   n_iterations;

	IterationStats(float error_, int n_iterations_):
		error(error_), n_iterations(n_iterations_) {;}

};

float l_inf_diff(thrust::device_vector<float>& a,
				 thrust::device_vector<float>& b);

float two_norm(thrust::device_vector<float>& a,
			   thrust::device_vector<float>& b);

void copy_boundaries(thrust::device_vector<float>& A,
					 thrust::device_vector<float>& B,
					 int ny, int nx);

template<class T>
void save_vector(const T& vec, const std::string& file, const char sep=' ')
{

	std::ofstream ofile(file.c_str());
	for(size_t i = 0; i<vec.size(); i++)
		ofile << vec[i] << sep;
	ofile << std::endl;
	ofile.close();

}

#endif
