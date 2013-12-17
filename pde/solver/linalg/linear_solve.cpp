
#include "linear_solve.hpp"
#include "backend.hpp"


/*
 * Solve for multiple b at once
 */
int d_ge_linear_solve(double* A, double* x, double* b, int N, int Nb)
{
  int* ipiv = new int[N];

  // Copy 'b' into 'x', LAPACK overwrites this
  for(int i=0; i<N*Nb; i++)
    x[i] = b[i];

  int info;
  info = BACKEND_dgesv(N, Nb, A, N, ipiv, x, Nb);

  delete[] ipiv;

  return info;

}

/*
 * Solve for single b
 */
int d_ge_linear_solve(double* A, double* x, double* b, int N)
{

  d_ge_linear_solve(A, x, b, N, 1);

}
