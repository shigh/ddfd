
#include "backend.hpp"
#include "lapacke.h"

double frob_norm(double *A, int M, int N)
{

  double norm;
  norm = LAPACKE_dlange(LAPACK_ROW_MAJOR, 'f', M, N, A, N);

  return norm;

}

int BACKEND_dgesv(int n, int nrhs,
                  double* a, int lda, int* ipiv,
                  double* b, int ldb)
{
  return LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs,
                       a, lda, ipiv, b, ldb );

}

