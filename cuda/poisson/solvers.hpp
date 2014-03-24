
#ifndef __SOLVERS_H
#define __SOLVERS_H

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cusp/csr_matrix.h>
#include "cusp_poisson.hpp"

template<typename ValueType, class DeviceVector>
class Solver3DBase
{

protected:

	DeviceVector f;

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

	// void solve(thrust::host_vector<ValueType>& x_h)
	// {
	// 	thrust::device_vector<ValueType> x_d(x_h.size());
	// 	thrust::copy(x_h.begin(), x_h.end(), x_d.begin());
	// 	solve(x_d);
	// 	thrust::copy(x_d.begin(), x_d.end(), x_h.begin());
	// }

	virtual void solve(DeviceVector& x_d) = 0;

};


template<typename ValueType, class DeviceVector>
class PoissonSolver3DCUSP : public Solver3DBase<ValueType, DeviceVector>
{

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

		

	}


};


#endif

