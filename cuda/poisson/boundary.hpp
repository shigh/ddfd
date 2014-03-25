

#ifndef __BOUNDARY_H
#define __BOUNDARY_H

#include <vector>
#include <cstddef>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

enum Boundary
{
    North  = 1u << 0,
	South  = 1u << 1,
	West   = 1u << 2,
	East   = 1u << 3,
	Top    = 1u << 4,
	Bottom = 1u << 5
};


template<typename T, class Vector>
class BoundarySet
{

private:

	Vector north;
	Vector south;
	Vector west;
	Vector east;
	Vector top;
	Vector bottom;

public:

	const size_t nz;
	const size_t ny;
	const size_t nx;

	BoundarySet(size_t nz_, size_t ny_, size_t nx_):
		nz(nz_), ny(ny_), nx(nx_),
		north(nz*nx, 0), south(nz*nx, 0),
		west(nz*ny, 0),  east(nz*ny, 0),
		top(ny*nx, 0),   bottom(ny*nx, 0) {;}

};

template<typename T>
class HostBoundarySet: BoundarySet<T, thrust::host_vector<T> >
{

private:

	typedef BoundarySet<T, thrust::host_vector<T> > Parent;

public:
	
	HostBoundarySet(size_t N): Parent(N) {;}

};

template<typename T>
class DeviceBoundarySet: BoundarySet<T, thrust::device_vector<T> >
{

private:

	typedef BoundarySet<T, thrust::host_vector<T> > Parent;

public:
	
	DeviceBoundarySet(size_t N): Parent(N) {;}

};

#endif
