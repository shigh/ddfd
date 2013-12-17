
#include "backend.hpp"
#include "mkl.h"
#include "mkl_lapacke.h"

double frob_norm(double *A, int M, int N)
{

  double norm;
  double *work = NULL;

  const char norm_type = 'f';

  norm = dlange(&norm_type, &M, &N, A, &M, work);

  return norm;

}

int BACKEND_dgesv(int n, int nrhs,
                  double* a, int lda, int* ipiv,
                  double* b, int ldb)
{
  return LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, nrhs,
                       a, lda, ipiv, b, ldb );

}
