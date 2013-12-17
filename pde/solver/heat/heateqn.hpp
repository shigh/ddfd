
#ifndef __HEAT_EQN_H
#define __HEAT_EQN_H

#include "solver.hpp"

class ExplicitHeatEqn1D: public Solver
{
protected:
  double dx, dt;

public:

  ExplicitHeatEqn1D(double dx, double dt): dx(dx), dt(dt) {};

  /*
   * Using very simple 4 point explicit stencil
   */
  void one_step(double* from, double* to, int n_points);

  ~ExplicitHeatEqn1D() {};
};

class ImplicitHeatEqn1D: public Solver
{
protected:
  double dx, dt;
  // Size of one time slice
  int nx;
  // Solution matrix - dense
  double* A;

public:

  ImplicitHeatEqn1D(int nx, double dx, double dt);

  /*
   * A hack using lapacke linear solve (dgesv)
   */
  void one_step(double* from, double* to, int n_points);

  ~ImplicitHeatEqn1D();
 
};

/*
 * 2D jacobi iteration
 */
class ImplicitHeatEqn2D: public Solver
{
protected:
  double dx, dy, dt;
  // Size of one time slice
  int nx, ny;

  double tol;
  int max_iter; 

  // The cython wrappers complain about non-static
  // data member initilizers
  void init_settings() { tol=0.000001; max_iter = 500; }

public:

  ImplicitHeatEqn2D(int nx, int ny, double dx, double dy, double dt);

  /*
   * Jacb Implicit
   */
  void one_step(double* from, double* to, int n_points);

  /*
   * Set iteration params
   */
  void set_tol(double tol) { this->tol = tol; }
  void set_max_iter(double max_iter) { this->max_iter = max_iter; }

};

class ImplicitHeatEqn3D: public Solver
{
protected:
  double dx, dy, dz, dt;

  int nx, ny, nz;

  double tol;
  int max_iter; 

  // The cython wrappers complain about non-static
  // data member initilizers
  void init_settings() { tol=0.000001; max_iter = 500; }

public:

  ImplicitHeatEqn3D(int nx, int ny, int nz, double dx, double dy, double dz, double dt);

  /*
   * Jacb Implicit
   */
  void one_step(double* from, double* to, int n_points);

  /*
   * Set iteration params
   */
  void set_tol(double tol) { this->tol = tol; }
  void set_max_iter(double max_iter) { this->max_iter = max_iter; }

};


class ExplicitHeatEqn2D: public Solver
{
protected:
  int nx, ny;
  double dx, dy, dt;

public:

  ExplicitHeatEqn2D(int nx, int ny, double dx, double dy, double dt):
    nx(nx), ny(ny), dx(dx), dy(dy), dt(dt) {};

  /*
   * Lie-Trotter operator splitting
   */
  void one_step(double* from, double* to, int n_points);

  ~ExplicitHeatEqn2D() {};
 
};

#endif
