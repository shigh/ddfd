
#ifndef __LINALG_BACKEND_H
#define __LINALG_BACKEND_H

double frob_norm(double *A, int M, int N);

int BACKEND_dgesv(int n, int nrhs,
                  double* a, int lda, int* ipiv,
		  double* b, int ldb);

#endif
