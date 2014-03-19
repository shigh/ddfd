/*
 * Run Jacobi examples
 */

#include <stdio.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include "utils.hpp"
#include "jacobi.hpp"


int main(void)
{
	int max_iter = 5000;
	float tol = 0.0001;
	float error;

	int nx = 100;
	int ny = 100;
	int nz = 100;
	float dx = 2*M_PI/(nx-1.);
	float dy = 2*M_PI/(ny-1.);
	float dz = 2*M_PI/(nz-1.);

	int N = nx*ny*nz;

	thrust::host_vector<float> x(N, 0);
	thrust::host_vector<float> b(N);

	thrust::device_vector<float> x_d(N);
	thrust::device_vector<float> b_d(N);
	
	for(int k=0; k<nz; k++)
		for(int i=0; i<ny; i++)
			for(int j=0; j<nx; j++)
				b[i+j*nx+k*nx*ny] = sin(j*dx)*sin(i*dy)*sin(dz);

	thrust::copy(x.begin(), x.end(), x_d.begin());
	thrust::copy(b.begin(), b.end(), b_d.begin());

	error = jacobi_solve_3d(x_d, b_d,
							nz, dz, ny, dy, nx, dx,
							max_iter, tol);
   
	// Copy from GPU
	thrust::copy(x_d.begin(), x_d.end(), x.begin());

	std::cout << "tol: "   << tol << " "
			  << "error: " << error << std::endl;
  
}
