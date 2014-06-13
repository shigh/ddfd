
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <math.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>
#include <thrust/device_vector.h>
#include "cusp_poisson.hpp"
#include "solvers.hpp"

BOOST_AUTO_TEST_SUITE( solver_cusp_3d_tests )

// Check for explosions
BOOST_AUTO_TEST_CASE( solver_cusp_constructor )
{

	int nx = 5;
	int ny = 10;
	int nz = 15;

	float dx, dy, dz;
	dx = dy = dz = 1;

	cusp::array1d<float, cusp::device_memory> b(nz*ny*nx, 0);

	PoissonSolver3DCUSP<float> solver(b, nz, dz, ny, dy, nx, dx);


}


BOOST_AUTO_TEST_CASE( solver_cusp_solver_completes )
{
	int nx, ny, nz;
	nx = ny = nz = 10;
	const int N = nx*ny*nz;

	float dx = 2*M_PI/(nx-1.);
	float dy = 2*M_PI/(ny-1.);
	float dz = 2*M_PI/(nz-1.);

	cusp::array1d<float, cusp::device_memory> b(N, 0);
	cusp::array1d<float, cusp::device_memory> x(N, 0);

	for(int k=0; k<nz; k++)
		for(int i=0; i<ny; i++)
			for(int j=0; j<nx; j++)
				b[j+i*nx+k*nx*ny] = sin(j*dx)*sin(i*dy)*sin(k*dz);


	PoissonSolver3DCUSP<float> solver(b, nz, dz, ny, dy, nx, dx);

	solver.solve(x);

}


BOOST_AUTO_TEST_SUITE_END()
