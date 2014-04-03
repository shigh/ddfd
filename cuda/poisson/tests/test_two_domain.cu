/*
 * Test two domain convergence
 */

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <vector>
#include <cstddef>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

#include "cusp_poisson.hpp"
#include "solvers.hpp"
#include "test_utils.hpp"

BOOST_AUTO_TEST_SUITE( two_domain_convg )


BOOST_AUTO_TEST_CASE( west_east )
{
	const int nx = 10;
	const int ny = nx;
	const int nz = ny;
	const int N = nx*ny*nz;

	const float dx = 2*M_PI/(nx-1.);
	const float dy = 2*M_PI/(ny-1.);
	const float dz = 2*M_PI/(nz-1.);

	// Region 1 has x<x2
	// Region 2 has x>=x1
	const int x1_bnd = 4;
	const int x2_bnd = 6;

	const int nx1 = x2_bnd;
	const int nx2 = nx-x1_bnd;

	// Offsets for boundary extraction
	const int off1 = x2_bnd-x1_bnd-1;
	const int off2 = off1;

	// Number of iterations
	const int n_iter = 10;

	// Reference solution	
	cusp::array1d<float, cusp::host_memory>   b_h(N, 0);
	cusp::array1d<float, cusp::device_memory> x_full(N, 0);

	for(int k=0; k<nz; k++)
		for(int i=0; i<ny; i++)
			for(int j=0; j<nx; j++)
				b_h[j+i*nx+k*nx*ny] = sin(j*dx)*sin(i*dy)*sin(k*dz);

	cusp::array1d<float, cusp::device_memory> b_full(b_h);
	PoissonSolver3DCUSP<float> solver_full(b_full, nz, dz, ny, dy, nx, dx);

	solver_full.solve(x_full);

	cusp::array1d<float, cusp::host_memory> x_full_h(x_full);

	
	// Left domain (domain 1)
	cusp::array1d<float, cusp::host_memory>   b1_h(nx1*ny*nz, 0);
	cusp::array1d<float, cusp::device_memory> x1(nx1*ny*nz, 0);

	for(int k=0; k<nz; k++)
		for(int i=0; i<ny; i++)
			for(int j=0; j<nx1; j++)
				b1_h[j+i*nx1+k*nx1*ny] = sin(j*dx)*sin(i*dy)*sin(k*dz);

	cusp::array1d<float, cusp::device_memory> b1(b1_h);
	PoissonSolver3DCUSP<float> solver1(b1, nz, dz, ny, dy, nx1, dx);

	// Right domain (domain 2)
	cusp::array1d<float, cusp::host_memory>   b2_h(nx2*ny*nz, 0);
	cusp::array1d<float, cusp::device_memory> x2(nx2*ny*nz, 0);

	for(int k=0; k<nz; k++)
		for(int i=0; i<ny; i++)
			for(int j=x1_bnd; j<nx; j++)
				b2_h[(j-x1_bnd)+i*nx2+k*nx2*ny] = sin(j*dx)*sin(i*dy)*sin(k*dz);

	cusp::array1d<float, cusp::device_memory> b2(b2_h);
	PoissonSolver3DCUSP<float> solver2(b2, nz, dz, ny, dy, nx2, dx);


	thrust::device_vector<float> tmp(nx*ny*nz, 0);
	float* tmp_ptr = thrust::raw_pointer_cast(&tmp[0]);
	for(int i=0; i<n_iter; i++)
	{
	    solver1.solve(x1);
	    solver2.solve(x2);

		extract_east<float>(thrust::raw_pointer_cast(&x1[0]),
							tmp_ptr, nz, ny, nx, off1);
		set_west<float>(tmp_ptr,
						thrust::raw_pointer_cast(&x2[0]),
						nz, ny, nx);

		extract_west<float>(thrust::raw_pointer_cast(&x2[0]),
							tmp_ptr, nz, ny, nx, off2);
		set_east<float>(tmp_ptr,
						thrust::raw_pointer_cast(&x1[0]),
						nz, ny, nx);

	    cusp::array1d<float, cusp::host_memory> x1_h(x1);
	    cusp::array1d<float, cusp::host_memory> x2_h(x2);

	    float error = 0;
	    for(int k=0; k<nz; k++)
			for(int i=0; i<ny; i++)
			{
				for(int j=0; j<nx1; j++)
					error+=square<float>(x_full_h[j+i*nx+k*nx*ny]-x1_h[j+i*nx1+k*nx1*ny]);

				for(int j=x1_bnd; j<nx; j++)
					error+=square<float>(x_full_h[j+i*nx+k*nx*ny]-x2_h[(j-x1_bnd)+i*nx2+k*nx2*ny]);

			}

	    std::cout << error << std::endl;

	}	
}


BOOST_AUTO_TEST_SUITE_END()

