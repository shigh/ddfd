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
template<typename T>
class BoundarySet
{

protected:

	T* north;
	T* south;
	T* west;
	T* east;
	T* top;
	T* bottom;

	virtual void allocate_buffer(T*& buf, size_t N) = 0;
	virtual void free_buffer(T*& buf) = 0;

	void init_all_buffers();
	void free_all_buffers();

public:

	const cudaMemoryType memory_space;
	const size_t nz;
	const size_t ny;
	const size_t nx;
	const size_t size_north;
	const size_t size_south;
	const size_t size_west;
	const size_t size_east;
	const size_t size_top;
	const size_t size_bottom;

	BoundarySet(size_t nz_, size_t ny_, size_t nx_, cudaMemoryType memory_space_);

	template<class BS>
	void copy(BS& from);

	T* get_north_ptr();
	T* get_south_ptr();
	T* get_west_ptr();
	T* get_east_ptr();
	T* get_top_ptr();
	T* get_bottom_ptr();

	virtual ~BoundarySet();

};

/*!
 * Boundaries stored on host
 */
template<typename T>
class HostBoundarySet: public BoundarySet<T>
{

private:

	void allocate_buffer(T*& buf, size_t N);
	void free_buffer(T*& buf);

public:

	HostBoundarySet(size_t nz_, size_t ny_, size_t nx_):
		BoundarySet<T>(nz_, ny_, nx_, cudaMemoryTypeHost)
	{this->init_all_buffers();}

	~HostBoundarySet() {this->free_all_buffers();}

};

/*!
 * Boundaries stored on device
 */
template<typename T>
class DeviceBoundarySet: public BoundarySet<T>
{

private:

	void allocate_buffer(T*& buf, size_t N);
	void free_buffer(T*& buf);

public:

	DeviceBoundarySet(size_t nz_, size_t ny_, size_t nx_):
		BoundarySet<T>(nz_, ny_, nx_, cudaMemoryTypeDevice) {this->init_all_buffers();}

	~DeviceBoundarySet() {this->free_all_buffers();}

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
void set_all_boundaries(DeviceBoundarySet<T>& bs, T* to_d,
						size_t nz, size_t ny, size_t nx);


/*!
 * Copy boundaries from 3D device matrix to BoundarySet object
 * \param from_d 3D device matrix to extract from
 * \param bs BoundarySet object to copy to
 */
template<typename T>
void extract_all_boundaries(T* from_d, DeviceBoundarySet<T>& bs,
							size_t nz, size_t ny, size_t nx);


/*!
 * Copy boundaries from 3D device matrix to BoundarySet object with offset
 * \param from_d 3D device matrix to extract from
 * \param bs BoundarySet object to copy to
 */
template<typename T>
void extract_all_boundaries(T* from_d, DeviceBoundarySet<T>& bs,
							size_t nz, size_t ny, size_t nx, size_t offset);


#include "boundary.inl"

