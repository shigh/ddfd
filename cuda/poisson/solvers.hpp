/*! \file
 * Classes encapsulating solver logic
 */

#ifndef __SOLVERS_H
#define __SOLVERS_H

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cusp/csr_matrix.h>
#include <cusp/krylov/cg.h>
#include "cusp_poisson.hpp"
#include "boundary.hpp"

/*!
 * Base class for solvers
 */
template<typename ValueType, class DeviceVector>
class Solver3DBase
{

protected:

	DeviceVector f;

	ValueType* device_pointer_cast_first(DeviceVector& vec)
	{
		return thrust::raw_pointer_cast(&vec[0]);
	}

public:

	typedef typename DeviceVector::size_type size_type;

	const size_type nz;
	const size_type ny;
	const size_type nx;

	const ValueType dz;
	const ValueType dy;
	const ValueType dx;

	Solver3DBase(DeviceVector& f_,
				 const size_type nz_, const ValueType dz_,
				 const size_type ny_, const ValueType dy_,
				 const size_type nx_, const ValueType dx_):
		f(f_), nz(nz_), dz(dz_),
		ny(ny_), dy(dy_), nx(nx_), dx(dx_) {};

	/*!
	 * Solve $A x_d = b$ for $x_d$
	 */
	virtual void solve(DeviceVector& x_d) = 0;

};

/*!
 * Solver built on CUSP sparse matrix library
 */
template<typename ValueType>
class PoissonSolver3DCUSP :
	public Solver3DBase<ValueType, cusp::array1d<ValueType, cusp::device_memory> >
{

public:

	typedef cusp::array1d<ValueType, cusp::device_memory> DeviceVector;
	using typename Solver3DBase<ValueType, DeviceVector>::size_type;
		
private:

	cusp::csr_matrix<size_type, ValueType, cusp::device_memory> A;

public:

	PoissonSolver3DCUSP(DeviceVector& f_,
						const size_type nz_, const ValueType dz_,
						const size_type ny_, const ValueType dy_,
						const size_type nx_, const ValueType dx_):
		Solver3DBase<ValueType, DeviceVector>(f_, nz_, dz_, ny_, dy_, nx_, dx_)
	{
        build_sparse_3d_poisson<size_type, ValueType>(A, this->nz, this->dz,
													  this->ny, this->dy,
													  this->nx, this->dx);		
    }


	void solve(DeviceVector& x_d)
	{
		set_all_boundaries(thrust::raw_pointer_cast(&x_d[0]),
						   thrust::raw_pointer_cast(&this->f[0]),
						   this->nz, this->ny, this->nx);
		cusp::default_monitor<float> monitor(this->f, 100, 1e-6);
		cusp::krylov::cg(A, x_d, this->f, monitor);	
	}


};


#endif

