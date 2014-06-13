/*! \file
 * Useful utility functions
 */

#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <fstream>
#include <cstddef>

/*! 
 * Keep track of key iteration stats.
 * Intended as a return value from solver type functions.
 */
struct IterationStats
{

	float error;
	int   n_iterations;

	IterationStats(float error_, int n_iterations_):
		error(error_), n_iterations(n_iterations_) {;}

};

/*! 
 * Max norm of the difference between a and b
 */
float l_inf_diff(thrust::device_vector<float>& a,
				 thrust::device_vector<float>& b);

/*! 
 * Two norm of the difference between a and b
 */
float two_norm(thrust::device_vector<float>& a,
			   thrust::device_vector<float>& b);

/*! 
 * Copy 2D boundaries from A to B
 * \param A 2D device_vector of size ny*nx to copy from
 * \param B 2D device_vector of size ny*nx to copy to
 */
void copy_boundaries(thrust::device_vector<float>& A,
					 thrust::device_vector<float>& B,
					 int ny, int nx);

/*! 
 * Save a vector to text file as a flat list of elements
 * \param vec Iterable vector
 * \param file Name of text file to save
 * \param sep Seperator for each element
 */
template<class T>
void save_vector(const T& vec, const std::string& file, const char sep=' ')
{

	std::ofstream ofile(file.c_str());
	for(size_t i = 0; i<vec.size(); i++)
		ofile << vec[i] << sep;
	ofile << std::endl;
	ofile.close();

}

