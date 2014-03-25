#include <math.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>
#include <thrust/device_vector.h>
#include "cusp_poisson.hpp"
#include "solvers.hpp"
#include "boundary.hpp"

int main(void)
{

	int nx, ny, nz;
	nx = ny = nz = 100;

	// float dx = 2*M_PI/(nx-1.);
	// float dy = 2*M_PI/(ny-1.);
	// float dz = 2*M_PI/(nz-1.);

	HostBoundarySet<float>   hbs(nz, ny, nx);
	DeviceBoundarySet<float> dbs(nz, ny, nx);

	hbs.copy(dbs);
	dbs.copy(hbs);

	return 0;
}
