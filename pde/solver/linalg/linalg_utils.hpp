
#ifndef __LINALG_UTILS_H
#define __LINALG_UTILS_H

double frob_norm(double *A, int M, int N);

double frob_norm_diff(double *A, double *B, int M, int N);

double frob_norm_diff_2(double *A, double *B, int N);

void transpose(double *A, double *B, int M, int N);

void copy_boundaries_2d(double *A, double *B, int M, int N);

#endif
