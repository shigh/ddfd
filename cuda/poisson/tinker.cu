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

	float* northh  = hbs.get_north_ptr();
	float* southh  = hbs.get_south_ptr();
	float* westh   = hbs.get_west_ptr();
	float* easth   = hbs.get_east_ptr();
	float* toph    = hbs.get_top_ptr();
	float* bottomh = hbs.get_bottom_ptr();

	float* northd  = dbs.get_north_ptr();
	float* southd  = dbs.get_south_ptr();
	float* westd   = dbs.get_west_ptr();
	float* eastd   = dbs.get_east_ptr();
	float* topd    = dbs.get_top_ptr();
	float* bottomd = dbs.get_bottom_ptr();


	return 0;
}
