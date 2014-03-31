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


BOOST_AUTO_TEST_SUITE( two_domain_convg )


BOOST_AUTO_TEST_CASE( west_east )
{
	const int nx = 100;
	const int ny = nx;
	const int nz = ny;
	const int N = nx*ny*nz;

	const float dx = 2*M_PI/(nx-1.);
	const float dy = 2*M_PI/(ny-1.);
	const float dz = 2*M_PI/(nz-1.);

	// Region 1 has x<x2
	// Region 2 has x>=x1
	const int x1_bnd = 45;
	const int x2_bnd = 55;

	const int nx1 = x2_bnd;
	const int nx2 = nx-x1_bnd;

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

	
	// Left domain (domain 1)
	cusp::array1d<float, cusp::host_memory>   b1_h(nx1*ny*nz, 0);
	cusp::array1d<float, cusp::device_memory> x1(nx1*ny*nz, 0);

	for(int k=0; k<nz; k++)
		for(int i=0; i<ny; i++)
			for(int j=0; j<nx1; j++)
				b1_h[j+i*nx1+k*nx1*ny] = sin(j*dx)*sin(i*dy)*sin(k*dz);

	cusp::array1d<float, cusp::device_memory> b1(b1_h);

	// Right domain (domain 2)
	cusp::array1d<float, cusp::host_memory>   b2_h(nx2*ny*nz, 0);
	cusp::array1d<float, cusp::device_memory> x2(nx2*ny*nz, 0);

	for(int k=0; k<nz; k++)
		for(int i=0; i<ny; i++)
			for(int j=x1_bnd; j<nx; j++)
				b2_h[(j-x1_bnd)+i*nx2+k*nx2*ny] = sin(j*dx)*sin(i*dy)*sin(k*dz);

	cusp::array1d<float, cusp::device_memory> b2(b2_h);
	


	
}


BOOST_AUTO_TEST_SUITE_END()

