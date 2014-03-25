

#ifndef __BOUNDARY_H
#define __BOUNDARY_H

#include <vector>
#include <cstddef>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

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

public:

	// These vector defs should be protected
	// I am keeping them public to simplify dev
	// Do not use them in calling code!!!
	Vector north;
	Vector south;
	Vector west;
	Vector east;
	Vector top;
	Vector bottom;


	const size_t nz;
	const size_t ny;
	const size_t nx;

	BoundarySet(size_t nz_, size_t ny_, size_t nx_):
		nz(nz_), ny(ny_), nx(nx_),
		north(nz_*nx_, 0), south(nz_*nx_, 0),
		west(nz_*ny_, 0),  east(nz_*ny_, 0),
		top(ny_*nx_, 0),   bottom(ny_*nx_, 0) {;}

	template<class FromBoundarySet>
	void copy(FromBoundarySet& from)
	{

		thrust::copy(from.north.begin(),  from.north.end(),  north.begin());
		thrust::copy(from.south.begin(),  from.south.end(),  south.begin());
		thrust::copy(from.west.begin(),   from.west.end(),   west.begin());
		thrust::copy(from.east.begin(),   from.east.end(),   east.begin());
		thrust::copy(from.top.begin(),    from.top.end(),    top.begin());
		thrust::copy(from.bottom.begin(), from.bottom.end(), bottom.begin());
				
	}

	// Get raw pointers for use as MPI comm buffers
	
	T* get_north_ptr()
	{
		return thrust::raw_pointer_cast(&north[0]);
	}

	T* get_south_ptr()
	{
		return thrust::raw_pointer_cast(&south[0]);
	}

	T* get_west_ptr()
	{
		return thrust::raw_pointer_cast(&west[0]);
	}

	T* get_east_ptr()
	{
		return thrust::raw_pointer_cast(&east[0]);
	}

	T* get_top_ptr()
	{
		return thrust::raw_pointer_cast(&top[0]);
	}

	T* get_bottom_ptr()
	{
		return thrust::raw_pointer_cast(&bottom[0]);
	}

};

template<typename T>
class HostBoundarySet: public BoundarySet<T, thrust::host_vector<T> >
{

private:

	typedef BoundarySet<T, thrust::host_vector<T> > Parent;

public:
	
	HostBoundarySet(size_t nz_, size_t ny_, size_t nx_): Parent(nz_,ny_,nx_) {;}

};

template<typename T>
class DeviceBoundarySet: public BoundarySet<T, thrust::device_vector<T> >
{

private:

	typedef BoundarySet<T, thrust::device_vector<T> > Parent;

public:
	
	DeviceBoundarySet(size_t nz_, size_t ny_, size_t nx_): Parent(nz_,ny_,nx_) {;}

};

#endif
