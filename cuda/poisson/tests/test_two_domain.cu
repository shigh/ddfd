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


	float error;
	thrust::device_vector<float> tmp(nx*ny*nz, 0);
	float* tmp_ptr = thrust::raw_pointer_cast(&tmp[0]);
	for(int i=0; i<n_iter; i++)
	{
	    solver1.solve(x1);
	    solver2.solve(x2);

		extract_east<float>(thrust::raw_pointer_cast(&x1[0]),
							tmp_ptr, nz, ny, nx1, off1);
		set_west<float>(tmp_ptr,
						thrust::raw_pointer_cast(&x2[0]),
						nz, ny, nx2);

		extract_west<float>(thrust::raw_pointer_cast(&x2[0]),
							tmp_ptr, nz, ny, nx2, off2);
		set_east<float>(tmp_ptr,
						thrust::raw_pointer_cast(&x1[0]),
						nz, ny, nx1);

	    cusp::array1d<float, cusp::host_memory> x1_h(x1);
	    cusp::array1d<float, cusp::host_memory> x2_h(x2);

		error = 0;
	    for(int k=0; k<nz; k++)
			for(int i=0; i<ny; i++)
			{

				for(int j=0; j<nx1; j++)
					error+=square<float>(x_full_h[j+i*nx+k*nx*ny]-x1_h[j+i*nx1+k*nx1*ny]);

				for(int j=x1_bnd; j<nx; j++)
					error+=square<float>(x_full_h[j+i*nx+k*nx*ny]-x2_h[(j-x1_bnd)+i*nx2+k*nx2*ny]);

			}

	}

	BOOST_CHECK( error < 10e-6 );	

}

BOOST_AUTO_TEST_CASE( north_south )
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
	const int y1_bnd = 4;
	const int y2_bnd = 6;

	const int ny1 = y2_bnd;
	const int ny2 = ny-y1_bnd;

	// Offsets for boundary extraction
	const int off1 = y2_bnd-y1_bnd-1;
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
	cusp::array1d<float, cusp::host_memory>   b1_h(nx*ny1*nz, 0);
	cusp::array1d<float, cusp::device_memory> x1(nx*ny1*nz, 0);

	for(int k=0; k<nz; k++)
		for(int i=0; i<ny1; i++)
			for(int j=0; j<nx; j++)
				b1_h[j+i*nx+k*nx*ny1] = sin(j*dx)*sin(i*dy)*sin(k*dz);

	cusp::array1d<float, cusp::device_memory> b1(b1_h);
	PoissonSolver3DCUSP<float> solver1(b1, nz, dz, ny1, dy, nx, dx);

	// Right domain (domain 2)
	cusp::array1d<float, cusp::host_memory>   b2_h(nx*ny2*nz, 0);
	cusp::array1d<float, cusp::device_memory> x2(nx*ny2*nz, 0);

	for(int k=0; k<nz; k++)
		for(int i=y1_bnd; i<ny; i++)
			for(int j=0; j<nx; j++)
				b2_h[j+(i-y1_bnd)*nx+k*nx*ny2] = sin(j*dx)*sin(i*dy)*sin(k*dz);

	cusp::array1d<float, cusp::device_memory> b2(b2_h);
	PoissonSolver3DCUSP<float> solver2(b2, nz, dz, ny2, dy, nx, dx);


	float error;
	thrust::device_vector<float> tmp(nx*ny*nz, 0);
	float* tmp_ptr = thrust::raw_pointer_cast(&tmp[0]);
	for(int i=0; i<n_iter; i++)
	{
	    solver1.solve(x1);
	    solver2.solve(x2);

		extract_north<float>(thrust::raw_pointer_cast(&x1[0]),
							 tmp_ptr, nz, ny1, nx, off1);
		set_south<float>(tmp_ptr,
						 thrust::raw_pointer_cast(&x2[0]),
						 nz, ny2, nx);

		extract_south<float>(thrust::raw_pointer_cast(&x2[0]),
							 tmp_ptr, nz, ny2, nx, off2);
		set_north<float>(tmp_ptr,
						 thrust::raw_pointer_cast(&x1[0]),
						 nz, ny1, nx);

	    cusp::array1d<float, cusp::host_memory> x1_h(x1);
	    cusp::array1d<float, cusp::host_memory> x2_h(x2);

		error = 0;
	    for(int k=0; k<nz; k++)
			for(int j=0; j<nx; j++)
			{

				for(int i=0; i<ny1; i++)
					error+=square<float>(x_full_h[j+i*nx+k*nx*ny]-x1_h[j+i*nx+k*nx*ny1]);

				for(int i=y1_bnd; i<ny; i++)
					error+=square<float>(x_full_h[j+i*nx+k*nx*ny]-x2_h[j+(i-y1_bnd)*nx+k*nx*ny2]);

			}

	}

	BOOST_CHECK( error < 10e-6 );	

}


BOOST_AUTO_TEST_SUITE_END()

