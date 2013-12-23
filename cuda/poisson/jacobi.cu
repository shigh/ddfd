
#include <stdio.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/extrema.h>
#include "utils.hpp"

/*
 * 2D jacobi iteration
 */
__global__
void jacobi_2d(float *x_d, float *xnew_d, float *b_d,
	       int ny, float dy, int nx, float dx)
{

  int x   = threadIdx.x + blockIdx.x*blockDim.x;
  int y   = threadIdx.y + blockIdx.y*blockDim.y;
  int x0  = x;
  int tid, north, south, east, west;


  while(y < ny)
    {
      while(x < nx)
	{

	  tid   = x + y*nx;
	  north = tid + nx;
	  south = tid - nx;
	  west  = tid - 1;
	  east  = tid + 1;

	  if( x>0 && x<nx-1 && y>0 && y<ny-1)
	    xnew_d[tid] = ((dx*dx)*b[tid]-x[north]-x[south]
			   -x[west]-x[east])/(-4.0);

	  x += blockIdx.x;

	}

      x = x0;
      y += blockDim.y;

    }

}

void call_jacobi_step(thrust::device_vector<float>& x,
		      thrust::device_vector<float>& xnew,
		      thrust::device_vector<float>& b,
		      int ny, float dy, int nx, float dx)
{

  dim3 dB(32, 32);
  dim3 dT(16, 16);

  //call Jacobi step
  jacobi_2d<<<dB, dT>>>(thrust::raw_pointer_cast(&b[0]),
			thrust::raw_pointer_cast(&x[0]),
			thrust::raw_pointer_cast(&xnew[0]),
			ny, dy, nx, dx);
  
}

float jacobi(thrust::device_vector<float>& x,
	    thrust::device_vector<float>& b,
	    int ny, float dy, int nx, float dx,
	    int max_iter, float tol)
{
   
  thrust::device_vector<float> xnew(nx*ny, 0);
  copy_boundaries(x, xnew, nx, ny);

  // Set to -1 to keep the iteration count correct
  int i = -1;
  // Init error >tol to avoid tripping condition in while
  // before the first iteration
  float error = tol + 1;

  while( error > tol && i < max_iter )
    {

      i++;
      
      //jacobi step
      if( i%2==0 )
	call_jacobi_step(x, xnew, b, ny, dy, nx, dx);
      else
	call_jacobi_step(xnew, x, b, ny, dy, nx, dx);

      //l_infinity norm 
      error = l_inf_diff(x, xnew);

    }

  if( i%2==0 )
    thrust::copy(xnew.begin(), xnew.end(), x.begin());

  return error;

}

// int main(void)
// {
//   int i;
//   int max_cnt=100;
//   float a=0;
//   float b=1;
//   int nx=40, ny=40;
//   int N = nx*ny;
//   float dx=(b-a)/((float) nx*ny-1);
//   float RHB=3.0;
//   float LHB=1.0;
//   float tol=1.0e-6;

//   thrust::host_vector<float> xn(N);
//   thrust::host_vector<float> rhs(N);

//   thrust::device_vector<float> xn_d(N);
//   thrust::device_vector<float> rhs_d(N);

//   for(i=0;i<nx*ny;i++){
//     rhs[i] = 0.0;
//     xn[i]  = 2.0;//((RHB-LHB)/(b-a))*(x-a)+LHB;
//   };

//   //set boundary conditions for poisson solve
//   for(i=0; i<nx; i++)
//     {
//       xn[i]           = LHB;
//       xn[nx*(ny-1)+i] = RHB;
//     }

//   for(i=0; i<ny; i++)
//     {
//       xn[nx*i]        = LHB;
//       xn[(nx+1)*i]    = RHB;
//     }

//   thrust::copy(xn.begin(), xn.end(), xn_d.begin());
//   thrust::copy(rhs.begin(), rhs.end(), rhs_d.begin());

//   jacobi(tol, dx, max_cnt, nx, ny, rhs_d, xn_d);
   
//   // Copy from GPU
//   thrust::copy(xn_d.begin(), xn_d.end(), xn.begin());

// }
