#include <cusp/coo_matrix.h>
#include <cusp/ell_matrix.h>
#include <cusp/print.h>

int main(void)
{

    // initialize matrix entries on host
	int nx, ny, nz;
	nx = ny = nz = 4;
	float dx, dy, dz;
	dx = dy = dz = .1;

	// allocate storage for (4,3) matrix with 6 nonzeros
	const int N = nx*ny*nz;
	const int boundary_count = 2*(nx*ny+nx*nz+ny*nz)-4*(nx+ny+nz)+8;
	const int interior_count = (nx-2)*(ny-2)*(nz-2);
	const int nonzero_count  = 1*boundary_count + 7*interior_count;
    cusp::coo_matrix<int, float, cusp::host_memory> A(N, N, nonzero_count);

	int m;
	int ind=0; // Running index of coo_matrix
	//int bc=0, ic=0;
	const float idx2 = 1/(dx*dx);
	const float idy2 = 1/(dy*dy);
	const float idz2 = 1/(dz*dz);
	const float af   = -2*(idx2 + idy2 + idz2);

	for(int k=0; k<nz; k++)
		for(int j=0; j<ny; j++)
			for(int i=0; i<nx; i++)
			{
				m = i + j*nx + k*nx*ny;
				if(i>0 && i<(nx-1) &&
				   j>0 && j<(ny-1) &&
				   k>0 && k<(nz-1))
				{
					A.row_indices[ind]    = m;
					A.column_indices[ind] = m;
					A.values[ind]         = af;
	
					A.row_indices[++ind]  = m;
					A.column_indices[ind] = m-1;
					A.values[ind]         = idx2;
	
					A.row_indices[++ind]  = m;
					A.column_indices[ind] = m+1;
					A.values[ind]         = idx2;

					A.row_indices[++ind]  = m;
					A.column_indices[ind] = m-nx;
					A.values[ind]         = idy2;
	
					A.row_indices[++ind]  = m;
					A.column_indices[ind] = m+nx;
					A.values[ind]         = idy2;

					A.row_indices[++ind]  = m;
					A.column_indices[ind] = m-nx*ny;
					A.values[ind]         = idz2;
	
					A.row_indices[++ind]  = m;
					A.column_indices[ind] = m+nx*ny;
					A.values[ind]         = idz2;
					
					//ic++;
				}
				else
				{
					A.row_indices[ind]    = m;
					A.column_indices[ind] = m;
					A.values[ind]         = 1;
					//bc++;
				}

				ind++;
			}

	//std::cout << ic << ' ' << interior_count << std::endl;
	//std::cout << bc << ' ' << boundary_count << std::endl;
    // print matrix entries
    //cusp::print(A);

	cusp::ell_matrix<int, float, cusp::device_memory> B(A);

    return 0;
}

