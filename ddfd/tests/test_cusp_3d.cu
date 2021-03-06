
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <math.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>
#include "cusp_poisson.hpp"


BOOST_AUTO_TEST_SUITE( cusp_3d_tests )

// Check for explosions
BOOST_AUTO_TEST_CASE( build_sparse_completes )
{

	int nx = 5;
	int ny = 10;
	int nz = 15;

	float dx, dy, dz;
	dx = dy = dz = 1;

	cusp::csr_matrix<int, float, cusp::device_memory> A;
	build_sparse_3d_poisson<int, float>(A, nz, dz, ny, dy, nx, dx);

}

// Make sure we can get the cg krylov solver to converge
BOOST_AUTO_TEST_CASE( solver_converges )
{

	int nx, ny, nz;
	nx = ny = nz = 50;

	float dx = 2*M_PI/(nx-1.);
	float dy = 2*M_PI/(ny-1.);
	float dz = 2*M_PI/(nz-1.);

	cusp::csr_matrix<int, float, cusp::device_memory> A;
	build_sparse_3d_poisson<int, float>(A, nz, dz, ny, dy, nx, dx);

	cusp::array1d<float, cusp::device_memory> b(A.num_rows, 0);
	cusp::array1d<float, cusp::device_memory> x(A.num_rows, 0);

	for(int k=0; k<nz; k++)
		for(int i=0; i<ny; i++)
			for(int j=0; j<nx; j++)
				b[j+i*nx+k*nx*ny] = sin(j*dx)*sin(i*dy)*sin(k*dz);

	cusp::default_monitor<float> monitor(b, 100, 1e-6);
	cusp::krylov::cg(A, x, b, monitor);

	BOOST_CHECK( monitor.converged() );

}

BOOST_AUTO_TEST_SUITE_END()
 
