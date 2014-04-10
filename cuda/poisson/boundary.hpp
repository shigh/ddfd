/*! \file 
 * Tools for working with matrix boundaries
 */

#ifndef __BOUNDARY_H
#define __BOUNDARY_H

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
 * Copy Top boundary from 3D matrix
 *
 * \param from_d 3D device matrix
 * \param to_d 2D device matrix
 * \param offset Number of steps back from Top
 */
template<typename T>
void extract_top(T* from_d, T* to_d,
				 size_t nz, size_t ny, size_t nx, size_t offset=0)
{

	cudaMemcpy(to_d, &from_d[(nz-1)*nx*ny - offset*nx*ny],
			   nx*ny*sizeof(T), cudaMemcpyDeviceToDevice);

}

/*!
 * Copy Top boundary into 3D matrix
 *
 * \param from_d 2D device matrix
 * \param to_d 3D device matrix
 */
template<typename T>
void set_top(T* from_d, T* to_d,
			 size_t nz, size_t ny, size_t nx)
{

	cudaMemcpy(&to_d[(nz-1)*nx*ny], from_d,
			   nx*ny*sizeof(T), cudaMemcpyDeviceToDevice);

}

/*!
 * Copy Bottom boundary from 3D matrix
 *
 * \param from_d 3D device matrix
 * \param to_d 2D device matrix
 * \param offset Number of steps back from Bottom
 */
template<typename T>
void extract_bottom(T* from_d, T* to_d,
					size_t nz, size_t ny, size_t nx, size_t offset=0)
{

	cudaMemcpy(to_d, &from_d[offset*nx*ny], nx*ny*sizeof(T), cudaMemcpyDeviceToDevice);

}

/*!
 * Copy Bottom boundary into 3D matrix
 *
 * \param from_d 2D device matrix
 * \param to_d 3D device matrix
 */
template<typename T>
void set_bottom(T* from_d, T* to_d,
					size_t nz, size_t ny, size_t nx)
{

	cudaMemcpy(to_d, from_d, nx*ny*sizeof(T), cudaMemcpyDeviceToDevice);

}


template<typename T>
__global__ void extract_west_kernel(T* from, T* to,
									size_t nz, size_t ny, size_t nx, size_t offset)
{

	int y = threadIdx.x + blockIdx.x*blockDim.x;
	int z = threadIdx.y + blockIdx.y*blockDim.y;
	int y_stride = blockDim.x;
	int z_stride = blockDim.y;
	int y0 = y;

	while(z < nz)
	{

		while(y < ny)
		{
			to[y + z*ny] = from[y*nx + z*nx*ny + offset];
			y += y_stride;
		}

		z += z_stride;
		y = y0;
	}

}


template<typename T>
__global__ void set_west_kernel(T* from, T* to,
								size_t nz, size_t ny, size_t nx)
{

	int y = threadIdx.x + blockIdx.x*blockDim.x;
	int z = threadIdx.y + blockIdx.y*blockDim.y;
	int y_stride = blockDim.x;
	int z_stride = blockDim.y;
	int y0 = y;

	while(z < nz)
	{

		while(y < ny)
		{
			to[y*nx + z*nx*ny] = from[y + z*ny];
			y += y_stride;
		}

		z += z_stride;
		y = y0;
	}

}

/*!
 * Copy West boundary from 3D matrix
 *
 * \param from_d 3D device matrix
 * \param to_d 2D device matrix
 * \param offset Number of steps back from West
 */
template<typename T>
void extract_west(T* from, T* to,
				  size_t nz, size_t ny, size_t nx, size_t offset=0)
{

	dim3 blocks(32, 32);
	dim3 threads(16, 16, 1);
	extract_west_kernel<<<blocks, threads>>>(from, to, nz, ny, nx, offset);

}

/*!
 * Copy West boundary into 3D matrix
 *
 * \param from_d 2D device matrix
 * \param to_d 3D device matrix
 */
template<typename T>
void set_west(T* from, T* to,
			  size_t nz, size_t ny, size_t nx)
{

	dim3 blocks(32, 32);
	dim3 threads(16, 16, 1);
	set_west_kernel<<<blocks, threads>>>(from, to, nz, ny, nx);

}


template<typename T>
__global__ void extract_east_kernel(T* from, T* to,
									size_t nz, size_t ny, size_t nx, size_t offset)
{

	int y = threadIdx.x + blockIdx.x*blockDim.x;
	int z = threadIdx.y + blockIdx.y*blockDim.y;
	int y_stride = blockDim.x;
	int z_stride = blockDim.y;
	int y0 = y;

	while(z < nz)
	{

		while(y < ny)
		{
			to[y + z*ny] = from[(nx-1) + y*nx + z*nx*ny - offset];
			y += y_stride;
		}

		z += z_stride;
		y = y0;
	}

}


template<typename T>
__global__ void set_east_kernel(T* from, T* to,
								size_t nz, size_t ny, size_t nx)
{

	int y = threadIdx.x + blockIdx.x*blockDim.x;
	int z = threadIdx.y + blockIdx.y*blockDim.y;
	int y_stride = blockDim.x;
	int z_stride = blockDim.y;
	int y0 = y;

	while(z < nz)
	{

		while(y < ny)
		{
			to[(nx-1) + y*nx + z*nx*ny] = from[y + z*ny];
			y += y_stride;
		}

		z += z_stride;
		y = y0;
	}

}

/*!
 * Copy East boundary from 3D matrix
 *
 * \param from_d 3D device matrix
 * \param to_d 2D device matrix
 * \param offset Number of steps back from East
 */
template<typename T>
void extract_east(T* from, T* to,
				  size_t nz, size_t ny, size_t nx, size_t offset=0)
{

	dim3 blocks(32, 32);
	dim3 threads(16, 16, 1);
	extract_east_kernel<<<blocks, threads>>>(from, to, nz, ny, nx, offset);

}

/*!
 * Copy East boundary into 3D matrix
 *
 * \param from_d 2D device matrix
 * \param to_d 3D device matrix
 */
template<typename T>
void set_east(T* from, T* to,
			  size_t nz, size_t ny, size_t nx)
{

	dim3 blocks(32, 32);
	dim3 threads(16, 16, 1);
	set_east_kernel<<<blocks, threads>>>(from, to, nz, ny, nx);

}


template<typename T>
__global__ void extract_north_kernel(T* from, T* to,
									 size_t nz, size_t ny, size_t nx, size_t offset)
{

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int z = threadIdx.y + blockIdx.y*blockDim.y;
	int x_stride = blockDim.x;
	int z_stride = blockDim.y;
	int x0 = x;

	while(z < nz)
	{

		while(x < nx)
		{
			to[x + z*nx] = from[(ny-1)*nx + x + z*nx*ny - offset*nx];
			x += x_stride;
		}

		z += z_stride;
		x = x0;
	}

}


template<typename T>
__global__ void set_north_kernel(T* from, T* to,
								 size_t nz, size_t ny, size_t nx)
{

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int z = threadIdx.y + blockIdx.y*blockDim.y;
	int x_stride = blockDim.x;
	int z_stride = blockDim.y;
	int x0 = x;

	while(z < nz)
	{

		while(x < nx)
		{
			to[(ny-1)*nx + x + z*nx*ny] = from[x + z*nx];
			x += x_stride;
		}

		z += z_stride;
		x = x0;
	}

}

/*!
 * Copy North boundary from 3D matrix
 *
 * \param from_d 3D device matrix
 * \param to_d 2D device matrix
 * \param offset Number of steps back from North
 */
template<typename T>
void extract_north(T* from, T* to,
				   size_t nz, size_t ny, size_t nx, size_t offset=0)
{

	dim3 blocks(32, 32);
	dim3 threads(16, 16, 1);
	extract_north_kernel<<<blocks, threads>>>(from, to, nz, ny, nx, offset);

}

/*!
 * Copy North boundary into 3D matrix
 *
 * \param from_d 2D device matrix
 * \param to_d 3D device matrix
 */
template<typename T>
void set_north(T* from, T* to,
			   size_t nz, size_t ny, size_t nx)
{

	dim3 blocks(32, 32);
	dim3 threads(16, 16, 1);
	set_north_kernel<<<blocks, threads>>>(from, to, nz, ny, nx);

}


template<typename T>
__global__ void extract_south_kernel(T* from, T* to,
									 size_t nz, size_t ny, size_t nx, size_t offset)
{

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int z = threadIdx.y + blockIdx.y*blockDim.y;
	int x_stride = blockDim.x;
	int z_stride = blockDim.y;
	int x0 = x;

	while(z < nz)
	{

		while(x < nx)
		{
			to[x + z*nx] = from[x + z*nx*ny + offset*nx];
			x += x_stride;
		}

		z += z_stride;
		x = x0;
	}

}


template<typename T>
__global__ void set_south_kernel(T* from, T* to,
								 size_t nz, size_t ny, size_t nx)
{

	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int z = threadIdx.y + blockIdx.y*blockDim.y;
	int x_stride = blockDim.x;
	int z_stride = blockDim.y;
	int x0 = x;

	while(z < nz)
	{

		while(x < nx)
		{
			to[x + z*nx*ny] = from[x + z*nx];
			x += x_stride;
		}

		z += z_stride;
		x = x0;
	}

}

/*!
 * Copy South boundary from 3D matrix
 *
 * \param from_d 3D device matrix
 * \param to_d 2D device matrix
 * \param offset Number of steps back from South
 */
template<typename T>
void extract_south(T* from, T* to,
				   size_t nz, size_t ny, size_t nx, size_t offset=0)
{

	dim3 blocks(32, 32);
	dim3 threads(16, 16, 1);
	extract_south_kernel<<<blocks, threads>>>(from, to, nz, ny, nx, offset);

}

/*!
 * Copy South boundary into 3D matrix
 *
 * \param from_d 2D device matrix
 * \param to_d 3D device matrix
 */
template<typename T>
void set_south(T* from, T* to,
			   size_t nz, size_t ny, size_t nx)
{

	dim3 blocks(32, 32);
	dim3 threads(16, 16, 1);
	set_south_kernel<<<blocks, threads>>>(from, to, nz, ny, nx);

}

/*!
 * Copy boundaries from 3D device matrix to 3D device matrix
 * \param from_d 3D device matrix to copy from
 * \param to_d 3D device matrix to copy to
 */
template<typename T>
void set_all_boundaries(T* from_d, T* to_d, size_t nz, size_t ny, size_t nx)
{

	const size_t maxn = std::max(std::max(nz, ny), nx);
	const size_t N    = maxn*maxn;
	thrust::device_vector<T> tmp(N);
	T* tmp_ptr = thrust::raw_pointer_cast(&tmp[0]);

	extract_top<T>(from_d, tmp_ptr, nz, ny, nx);
	set_top<T>(tmp_ptr, to_d, nz, ny, nx);

	extract_bottom<T>(from_d, tmp_ptr, nz, ny, nx);
	set_bottom<T>(tmp_ptr, to_d, nz, ny, nx);

	extract_north<T>(from_d, tmp_ptr, nz, ny, nx);
	set_north<T>(tmp_ptr, to_d, nz, ny, nx);

	extract_south<T>(from_d, tmp_ptr, nz, ny, nx);
	set_south<T>(tmp_ptr, to_d, nz, ny, nx);

	extract_west<T>(from_d, tmp_ptr, nz, ny, nx);
	set_west<T>(tmp_ptr, to_d, nz, ny, nx);

	extract_east<T>(from_d, tmp_ptr, nz, ny, nx);
	set_east<T>(tmp_ptr, to_d, nz, ny, nx);
	
}

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

	BoundarySet(size_t nz_, size_t ny_, size_t nx_):
		nz(nz_), ny(ny_), nx(nx_),
		north(nz_*nx_, 0), south(nz_*nx_, 0),
		west(nz_*ny_, 0),  east(nz_*ny_, 0),
		top(ny_*nx_, 0),   bottom(ny_*nx_, 0) {;}

	/*!
	 * Copy all boundaries from a boundary set.
	 * Intended for use moving boundaries to and from device/host.
	 */
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


#endif
