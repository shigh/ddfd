#ifndef __SOLVER_H
#define __SOLVER_H

/*
 * Abstract Base class for solution functions.
 */
class Solver
{
protected:

public:

  /*
   * Overwrite to using from as init values
   */
  virtual void one_step(double* from, double* to, int n_points) = 0;

  virtual ~Solver() = 0;

};

inline Solver::~Solver() {}

#endif
