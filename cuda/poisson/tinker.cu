#include <math.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>
#include <thrust/device_vector.h>
#include "cusp_poisson.hpp"
#include "solvers.hpp"

int main(void)
{

	int nx, ny, nz;
	nx = ny = nz = 100;

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


	PoissonSolver3DCUSP<float> solver(b, nz, dz, ny, dy, nx, dx);

	solver.solve(x);

	return 0;
}
