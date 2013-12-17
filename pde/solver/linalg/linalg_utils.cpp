
#include "linalg_utils.hpp"
#include "backend.hpp"
#include <cmath>

/*
 * Matrix/Vector subtraction 
 * C <- A - B
 */
void sub(double *A, double *B, double *C, int N)
{
  // Consider swapping in blas later
  // Thats why this is in a seperate funcion
  for(int i=0; i<N; i++)
    C[i] = A[i] - B[i];

}

/*
 * Frob norm of MxN matrices A - B
 */
double frob_norm_diff(double *A, double *B, int M, int N)
{

  int size = M*N;
  double norm;
  double *C = new double[size];

  sub(A, B, C, size);
  norm = frob_norm(C, M, N);

  delete[] C;

  return norm;

}

/*
 * Not using linalg backend
 */
double frob_norm_diff_2(double *A, double *B, int N)
{

  double sum = 0;
  double diff;
  for(int i=0; i<N; i++)
    {
      diff = A[i] - B[i];
      sum += diff*diff;
    }

  return std::sqrt(sum);

}


/*
 * Transpose MxN matrix A into NxM matrix B
 *
 * M = number of rows
 * N = number of cols
 * Assume both matrices are row major
 */
void transpose(double *A, double *B, int M, int N)
{

  for(int i=0; i<M; i++)
    for(int j=0; j<N; j++)
      B[j*M+i] = A[i*N+j];

}

/*
 * Copy the boundaries of A into the boundaries of B
 *
 * A = MxN matrix
 * B = MxN matrix
 * M = number of rows
 * N = number of cols
 */
void copy_boundaries_2d(double *A, double *B, int M, int N)
{

  // Copy top and bottom
  for(int i=0; i<N; i++)
    {
      B[i] = A[i];
      B[i+(M-1)*N] = A[i+(M-1)*N];
    }

  // Copy left and right
  for(int i=0; i<M; i++)
    {
      B[i*N] = A[i*N];
      B[i*N+N-1] = A[i*N+N-1];
    }

}

