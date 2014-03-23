#include <math.h>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>


template<typename IndexType, typename ValueType, class SparseMatrix>
void build_sparse_3d_poisson(SparseMatrix& A_out,
							 const IndexType nz, const ValueType dz,
							 const IndexType ny, const ValueType dy,
							 const IndexType nx, const ValueType dx)
{

	const IndexType N = nx*ny*nz;
	const IndexType boundary_count = 2*(nx*ny+nx*nz+ny*nz)-4*(nx+ny+nz)+8;
	const IndexType interior_count = (nx-2)*(ny-2)*(nz-2);
	const IndexType nonzero_count  = 1*boundary_count + 7*interior_count;
    cusp::coo_matrix<IndexType, ValueType, cusp::host_memory> A(N, N, nonzero_count);

	IndexType m;
	IndexType ind=0; // Running index of coo_matrix
	const ValueType idx2 = 1/(dx*dx);
	const ValueType idy2 = 1/(dy*dy);
	const ValueType idz2 = 1/(dz*dz);
	const ValueType af   = -2*(idx2 + idy2 + idz2);

	for(IndexType k=0; k<nz; k++)
		for(IndexType j=0; j<ny; j++)
			for(IndexType i=0; i<nx; i++)
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
					
				}
				else
				{
					A.row_indices[ind]    = m;
					A.column_indices[ind] = m;
					A.values[ind]         = 1;
				}

				ind++;
			}

	A_out = SparseMatrix(A);

}
							 


