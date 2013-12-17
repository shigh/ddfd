#include <algorithm>
#include <iostream>
#include "heateqn.hpp"
#include "linalg/linear_solve.hpp"
#include "linalg/linalg_utils.hpp"

void ExplicitHeatEqn1D::one_step(double* from, double* to, int n_points)
{

  double h = dt/(dx*dx);

  // Second diffs in space
  double* x2d = new double[n_points];

  // Calculate second diffs
  for(int i=1; i<n_points-1; i++)
    x2d[i] = from[i+1] - 2*from[i] + from[i-1];

  // Calculate to
  for(int i=1; i<n_points-1; i++)
    to[i] = from[i] + h*x2d[i];

  delete [] x2d;
}

void build_A(double* A, int N, double dx, double dt)
{
  double beta = dt/dx/dx;
  double diag = 1+2*beta;

  // Zero out A
  for(int i=0; i<N*N; i++)
    A[i] = 0.;

  for (int i=1; i<N-1; i++ )
    {
      A[i*N + i  ] =  diag;
      A[i*N + i+1] = -beta;
      A[i*N + i-1] = -beta;
    }

  A[0]     = 1;
  A[N*N-1] = 1;
}

ImplicitHeatEqn1D::ImplicitHeatEqn1D(int nx, double dx, double dt): dx(dx), dt(dt), nx(nx)
{
  A = new double[nx*nx];
  build_A(A, nx, dx, dt);
}

ImplicitHeatEqn1D::~ImplicitHeatEqn1D()
{
  delete[] A;
}

void ImplicitHeatEqn1D::one_step(double* from, double* to, int n_points)
{
  int info;
  build_A(A, nx, dx, dt);
  info = d_ge_linear_solve(A, to, from, nx);
}


ImplicitHeatEqn2D::ImplicitHeatEqn2D(int nx, int ny, double dx, double dy, double dt):
  dx(dx), dy(dy), dt(dt), nx(nx), ny(ny) { init_settings(); }


void ImplicitHeatEqn2D::one_step(double* from, double* to, int n_points)
{
  int N = nx*ny;
  double *xn   = new double[N];
  double *xnp1 = new double[N];

  double kx    = dt/(dx*dx);
  double ky    = dt/(dy*dy);
  double denom = 1. + 2*(kx+ky);

  for(int i=0; i<N; i++)
    {
      xn[i]   = to[i];
      xnp1[i] = to[i];
    }

  int center, top, bottom, left, right;
  int iter = 0;
  double norm = 1;
  while( norm>tol && iter<max_iter )
    {
      for(int i=1; i<ny-1; i++)
	for(int j=1; j<nx-1; j++)
	  {
	    center = i*nx + j;
	    top    = center + nx;
	    bottom = center - nx;
	    left   = center - 1;
	    right  = center + 1;

	    xnp1[center] = (from[center] +
			    kx*(xn[left] + xn[right]) +
			    ky*(xn[top]  + xn[bottom]))/denom;
	  }

      norm = frob_norm_diff(xn, xnp1, nx , ny);

      std::swap(xn, xnp1);

      iter++;
    }

  for(int i=0; i<N; i++)
    to[i] = xn[i];

  delete[] xn;
  delete[] xnp1;
  
}

ImplicitHeatEqn3D::ImplicitHeatEqn3D(int nx, int ny, int nz, double dx, double dy, double dz, double dt):
  dx(dx), dy(dy), dz(dz), dt(dt), nx(nx), ny(ny), nz(nz) { init_settings(); }


void ImplicitHeatEqn3D::one_step(double* from, double* to, int n_points)
{
  int N = nx*ny*nz;
  double *xn   = new double[N];
  double *xnp1 = new double[N];

  double kx    = dt/(dx*dx);
  double ky    = dt/(dy*dy);
  double kz    = dt/(dz*dz);
  double denom = 1. + 2*(kx+ky+kz);

  for(int i=0; i<N; i++)
    {
      xn[i]   = to[i];
      xnp1[i] = to[i];
    }

  int center, top, bottom, left, right, up, down;
  int iter = 0;
  double norm = 1;
  while( norm>tol && iter<max_iter )
    {
      for(int k=1; k<nz-1; k++)
	for(int i=1; i<ny-1; i++)
	  for(int j=1; j<nx-1; j++)
	    {
	      center = k*nx*ny + i*nx + j;
	      top    = center + nx;
	      bottom = center - nx;
	      left   = center - 1;
	      right  = center + 1;
	      up     = center + nx*ny;
	      down   = center - nx*ny;

	      xnp1[center] = (from[center] +
			      kx*(xn[left] + xn[right])  +
			      ky*(xn[top]  + xn[bottom]) +
			      kz*(xn[up]   + xn[down])
			      )/denom;
	    }

      norm = frob_norm_diff_2(xn, xnp1, N);

      std::swap(xn, xnp1);

      iter++;
    }

  for(int i=0; i<N; i++)
    to[i] = xn[i];

  delete[] xn;
  delete[] xnp1;
  
}

/*
 * Lie-Trotter splitting using 1D solver as a base
 * with standard 4-point explicit stencil
 */
void ExplicitHeatEqn2D::one_step(double* from, double* to, int n_points)
{

  ExplicitHeatEqn1D sol(dx, dt);

  // x direction
  for(int i=0; i<ny; i++)
    sol.one_step(from+i*nx, to+i*nx, nx);

  // y direction
  sol = ExplicitHeatEqn1D(dy, dt);
  double* y_vals = new double[ny];
  double* s_vals = new double[ny];

  for(int i=0; i<nx; i++)
    {
      // Get y_vals from to
      for(int j=0; j<ny; j++)
	y_vals[j] = to[j*nx+i];

      // Solve for y_vals
      sol.one_step(y_vals, s_vals, ny);

      // Put s_vals into to
      for(int j=0; j<ny; j++)
	to[j*nx+i] = s_vals[j];
      
    }

  delete [] y_vals;
  delete [] s_vals;

}
