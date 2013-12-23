#include "utils.hpp"
#include <stdio.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>


struct AbsDiff
{
  __host__ __device__
  float operator()(float a, float b)
  {
    return abs(a-b);
  }

};

float l_inf_diff(thrust::device_vector<float>& a,
 	         thrust::device_vector<float>& b)
{
  thrust::device_vector<float> tmp(a.size());
  thrust::transform(a.begin(), a.end(), b.begin(),
		   tmp.begin(), AbsDiff());

  thrust::device_vector<float>::iterator max_it =
    thrust::max_element(tmp.begin(), tmp.end());
  
  return *max_it;
}

struct SquareDiff
{
  __host__ __device__
  float operator()(float a, float b)
  {
    float k = a-b;
    return k*k;
  }

};


float two_norm(thrust::device_vector<float>& a,
 	       thrust::device_vector<float>& b)
{
  thrust::device_vector<float> tmp(a.size());
  thrust::transform(a.begin(), a.end(), b.begin(),
		    tmp.begin(), SquareDiff());

  float sum = thrust::reduce(tmp.begin(), tmp.end(), (float) 0, thrust::plus<float>());
  
  return std::sqrt(sum);
}


/*
 * Copy boundaries from A into B
 */
void copy_boundaries(thrust::device_vector<float>& A,
		     thrust::device_vector<float>& B,
		     int ny, int nx)
{

  // First and last rows
  thrust::copy_n(A.begin(), nx, B.begin());
  thrust::copy_n(&A[(ny-1)*nx], nx, &B[(ny-1)*nx]);

  // First and last cols
  for(int i=0; i<ny; i++)
    {
      B[i*nx]      = A[i*nx];
      B[i*nx+nx-1] = A[i*nx+nx-1];
    }

}
