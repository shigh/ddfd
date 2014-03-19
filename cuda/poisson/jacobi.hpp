
#ifndef __JACOBI_H
#define __JACOBI_H

#include <thrust/device_vector.h>

void call_jacobi_step(thrust::device_vector<float>& x,
					  thrust::device_vector<float>& xnew,
					  thrust::device_vector<float>& b,
					  int ny, float dy, int nx, float dx);

float jacobi_solve_2d(thrust::device_vector<float>& x,
					  thrust::device_vector<float>& b,
					  int ny, float dy, int nx, float dx,
					  int max_iter, float tol);

float jacobi_solve_3d(thrust::device_vector<float>& x,
					  thrust::device_vector<float>& b,
					  int nz, float dz, int ny, float dy, int nx, float dx,
					  int max_iter, float tol);


#endif
