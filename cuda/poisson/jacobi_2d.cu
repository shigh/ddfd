//-------------------------------------------
//
//MTH 995: Jacobi iteration on GPU for Poisson
//
//BY: Andrew Christlieb
//
// Concepts: Jacobi Iteration in CUDA 1D
//
//#include <iostream>
#include <stdio.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include "utils.hpp"

__global__ void jacobi_2d(float dx, float *b_d, float *x_d, float *xnew_d,
			       int nx, int ny)
{
  //compute initial index via flatening
  int x      = threadIdx.x + blockIdx.x*blockDim.x;
  int y      = threadIdx.y + blockIdx.y*blockDim.y;

  // index of values in underlying linear arrays
  int tid    = x + y*nx;
  int stride = nx;
  int top, bottom, left, right;

  top    = tid + stride;
  bottom = tid - stride;
  left   = tid - 1;
  right  = tid + 1;

  //one step of Jacobi for one iteration of alpaca operator
  if( x>0 && x<nx-1 && y>0 && y<ny-1)
    xnew_d[tid] = ((dx*dx)*b_d[tid]-x_d[top]-x_d[bottom]
		   -x_d[left]-x_d[right])/(-4.0);

}

void call_jacobi_step(float dx, thrust::device_vector<float>& b_d,
			        thrust::device_vector<float>& x_d,
			        thrust::device_vector<float>& xnew_d,
		      int nx, int ny)
{

  dim3 dB(32, 32);
  dim3 dT(16, 16);

  //call Jacobi step
  jacobi_2d<<<dB, dT>>>(dx, thrust::raw_pointer_cast(&b_d[0]),
			    thrust::raw_pointer_cast(&x_d[0]),
			    thrust::raw_pointer_cast(&xnew_d[0]),
			    nx, ny);
  
}

void jacobi(float tol, float dx, int max_cnt, int nx, int ny,
	    thrust::device_vector<float>& rhs_d,
	    thrust::device_vector<float>& xn_d)
{
   
  int i = 0;
  float max = 1;
  thrust::device_vector<float> xnp1_d(nx*ny, 0);
  float elapsed;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  copy_boundaries(xn_d, xnp1_d, nx, ny);

  while( max > tol && i<max_cnt){

    //jacobi step
    cudaEventRecord(start, 0);
    call_jacobi_step(dx, rhs_d, xn_d, xnp1_d, nx, ny);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    //l_infinity norm 
    max = l_inf_diff(xn_d, xnp1_d);

    //copy data from new to old
    thrust::copy(xnp1_d.begin(), xnp1_d.end(), xn_d.begin());

    i++;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("step %d - %f - %f\n",i,max,elapsed);

  };

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

int main(void)
{
  int i;                  
  int max_cnt=100;
  float a=0;
  float b=1;                   
  int nx=40, ny=40;
  int N = nx*ny;
  float dx=(b-a)/((float) nx*ny-1);
  float RHB=3.0;   
  float LHB=1.0;
  float tol=1.0e-6;

  thrust::host_vector<float> xn(N);
  thrust::host_vector<float> rhs(N);

  thrust::device_vector<float> xn_d(N);
  thrust::device_vector<float> rhs_d(N);

  for(i=0;i<nx*ny;i++){
    rhs[i] = 0.0;
    xn[i]  = 2.0;//((RHB-LHB)/(b-a))*(x-a)+LHB;
  };

  //set boundary conditions for poisson solve
  for(i=0; i<nx; i++)
    {
      xn[i]           = LHB;
      xn[nx*(ny-1)+i] = RHB;
    }

  for(i=0; i<ny; i++)
    {
      xn[nx*i]        = LHB;
      xn[(nx+1)*i]    = RHB;
    }

  thrust::copy(xn.begin(), xn.end(), xn_d.begin());
  thrust::copy(rhs.begin(), rhs.end(), rhs_d.begin());

  jacobi(tol, dx, max_cnt, nx, ny, rhs_d, xn_d);
   
  // Copy from GPU
  thrust::copy(xn_d.begin(), xn_d.end(), xn.begin());

}
