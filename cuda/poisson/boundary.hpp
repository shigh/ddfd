/*! \file 
 * Tools for working with matrix boundaries
 */

#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

/*!
 * For selecting boundaries
 */
enum Boundary
{
    North  = 1u << 0,
	South  = 1u << 1,
	West   = 1u << 2,
	East   = 1u << 3,
	Top    = 1u << 4,
	Bottom = 1u << 5
};


/*!
 * Hold and move all boundaries of a 3D matrix
 */
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

	BoundarySet(size_t nz_, size_t ny_, size_t nx_);

	/*!
	 * Copy all boundaries from a boundary set.
	 * Intended for use moving boundaries to and from device/host.
	 */
	template<class FromBoundarySet>
	void copy(FromBoundarySet& from);


	T* get_north_ptr();
	T* get_south_ptr();
	T* get_west_ptr();
	T* get_east_ptr();
	T* get_top_ptr();
	T* get_bottom_ptr();

};

/*!
 * Boundaries stored on host
 */
template<typename T>
class HostBoundarySet: public BoundarySet<T, thrust::host_vector<T> >
{

private:

	typedef BoundarySet<T, thrust::host_vector<T> > Parent;

public:
	
	HostBoundarySet(size_t nz_, size_t ny_, size_t nx_): Parent(nz_,ny_,nx_) {;}

};

/*!
 * Boundaries stored on device
 */
template<typename T>
class DeviceBoundarySet: public BoundarySet<T, thrust::device_vector<T> >
{

private:

	typedef BoundarySet<T, thrust::device_vector<T> > Parent;

public:
	
	DeviceBoundarySet(size_t nz_, size_t ny_, size_t nx_): Parent(nz_,ny_,nx_) {;}

};


/*!
 * Copy Top boundary from 3D matrix
 *
 * \param from_d 3D device matrix
 * \param to_d 2D device matrix
 * \param offset Number of steps back from Top
 */
template<typename T>
void extract_top(T* from_d, T* to_d,
				 size_t nz, size_t ny, size_t nx, size_t offset=0);


/*!
 * Copy Top boundary into 3D matrix
 *
 * \param from_d 2D device matrix
 * \param to_d 3D device matrix
 */
template<typename T>
void set_top(T* from_d, T* to_d,
			 size_t nz, size_t ny, size_t nx);


/*!
 * Copy Bottom boundary from 3D matrix
 *
 * \param from_d 3D device matrix
 * \param to_d 2D device matrix
 * \param offset Number of steps back from Bottom
 */
template<typename T>
void extract_bottom(T* from_d, T* to_d,
					size_t nz, size_t ny, size_t nx, size_t offset=0);


/*!
 * Copy Bottom boundary into 3D matrix
 *
 * \param from_d 2D device matrix
 * \param to_d 3D device matrix
 */
template<typename T>
void set_bottom(T* from_d, T* to_d,
				size_t nz, size_t ny, size_t nx);


/*!
 * Copy West boundary from 3D matrix
 *
 * \param from_d 3D device matrix
 * \param to_d 2D device matrix
 * \param offset Number of steps back from West
 */
template<typename T>
void extract_west(T* from, T* to,
				  size_t nz, size_t ny, size_t nx, size_t offset=0);


/*!
 * Copy West boundary into 3D matrix
 *
 * \param from_d 2D device matrix
 * \param to_d 3D device matrix
 */
template<typename T>
void set_west(T* from, T* to,
			  size_t nz, size_t ny, size_t nx);


/*!
 * Copy East boundary from 3D matrix
 *
 * \param from_d 3D device matrix
 * \param to_d 2D device matrix
 * \param offset Number of steps back from East
 */
template<typename T>
void extract_east(T* from, T* to,
				  size_t nz, size_t ny, size_t nx, size_t offset=0);


/*!
 * Copy East boundary into 3D matrix
 *
 * \param from_d 2D device matrix
 * \param to_d 3D device matrix
 */
template<typename T>
void set_east(T* from, T* to,
			  size_t nz, size_t ny, size_t nx);


/*!
 * Copy North boundary from 3D matrix
 *
 * \param from_d 3D device matrix
 * \param to_d 2D device matrix
 * \param offset Number of steps back from North
 */
template<typename T>
void extract_north(T* from, T* to,
				   size_t nz, size_t ny, size_t nx, size_t offset=0);


/*!
 * Copy North boundary into 3D matrix
 *
 * \param from_d 2D device matrix
 * \param to_d 3D device matrix
 */
template<typename T>
void set_north(T* from, T* to,
			   size_t nz, size_t ny, size_t nx);


/*!
 * Copy South boundary from 3D matrix
 *
 * \param from_d 3D device matrix
 * \param to_d 2D device matrix
 * \param offset Number of steps back from South
 */
template<typename T>
void extract_south(T* from, T* to,
				   size_t nz, size_t ny, size_t nx, size_t offset=0);


/*!
 * Copy South boundary into 3D matrix
 *
 * \param from_d 2D device matrix
 * \param to_d 3D device matrix
 */
template<typename T>
void set_south(T* from, T* to,
			   size_t nz, size_t ny, size_t nx);


/*!
 * Copy boundaries from 3D device matrix to 3D device matrix
 * \param from_d 3D device matrix to copy from
 * \param to_d 3D device matrix to copy to
 */
template<typename T>
void set_all_boundaries(T* from_d, T* to_d, size_t nz, size_t ny, size_t nx);


/*!
 * Copy boundaries from BoundarySet to 3D device matrix
 * \param bs DeviceBoundarySet object to copy from
 * \param to_d 3D device matrix to copy to
 */
template<typename T>
void set_all_boundaries(const DeviceBoundarySet<T>& bs, T* to_d,
						size_t nz, size_t ny, size_t nx);


/*!
 * Copy boundaries from 3D device matrix to BoundarySet object
 * \param from_d 3D device matrix to extract from
 * \param bs BoundarySet object to copy to
 */
template<typename T>
void extract_all_boundaries(const T* from_d, DeviceBoundarySet<T>& bs,
							size_t nz, size_t ny, size_t nx);


#include "boundary.inl"

