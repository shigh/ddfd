
#include <stdio.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include "utils.hpp"

/*
 * 2D jacobi iteration
 */
__global__
void jacobi_2d(float *x_d, float *xnew_d, float *b_d,
			   int ny, float dy, int nx, float dx)
{

	int x  = threadIdx.x + blockIdx.x*blockDim.x;
	int y  = threadIdx.y + blockIdx.y*blockDim.y;
	int x0 = x;
	int tid, north, south, east, west;

	float k  =(dx*dx*dy*dy)/(dx*dx + dy*dy);
	float kx = k/(dx*dx);
	float ky = k/(dy*dy);

	while(y < ny)
    {
		while(x < nx)
		{

			tid   = x + y*nx;
			north = tid + nx;
			south = tid - nx;
			west  = tid - 1;
			east  = tid + 1;

			if( x>0 && x<nx-1 && y>0 && y<ny-1)
				xnew_d[tid] = (k*b_d[tid] -
							   ky*(x_d[north] + x_d[south]) -
							   kx*(x_d[west]  + x_d[east]))/(-2.0);

			x += blockDim.x;

		}

		x = x0;
		y += blockDim.y;

    }

}

void call_jacobi_step_2d(thrust::device_vector<float>& x_d,
						 thrust::device_vector<float>& xnew_d,
						 thrust::device_vector<float>& b_d,
						 int ny, float dy, int nx, float dx)
{

	dim3 dB(32, 32);
	dim3 dT(16, 16);

	//call Jacobi step
	jacobi_2d<<<dB, dT>>>(thrust::raw_pointer_cast(&x_d[0]),
						  thrust::raw_pointer_cast(&xnew_d[0]),
						  thrust::raw_pointer_cast(&b_d[0]),
						  ny, dy, nx, dx);
  
}

float jacobi_solve_2d(thrust::device_vector<float>& x_d,
					  thrust::device_vector<float>& b_d,
					  int ny, float dy, int nx, float dx,
					  int max_iter, float tol)
{
   
	thrust::device_vector<float> xnew_d(nx*ny, 0);
	thrust::copy(x_d.begin(), x_d.end(), xnew_d.begin());
	//copy_boundaries(x, xnew, nx, ny);

	// Set to -1 to keep the iteration count correct
	int i = -1;
	// Init error >tol to avoid tripping condition in while
	// before the first iteration
	float error = tol + 1;

	while( error > tol && i < max_iter )
    {

		i++;
      
		//jacobi step
		if( i%2==0 )
			call_jacobi_step_2d(x_d, xnew_d, b_d, ny, dy, nx, dx);
		else
			call_jacobi_step_2d(xnew_d, x_d, b_d, ny, dy, nx, dx);

		//l_infinity norm 
		// error = l_inf_diff(x_d, xnew_d);
		error = two_norm(x_d, xnew_d); 

    }

	if( i%2==0 )
		thrust::copy(xnew_d.begin(), xnew_d.end(), x_d.begin());

	return error;

}

