
#include "boundary.hpp"


template<typename T>
void extract_top(T* from_d, T* to_d,
				 size_t nz, size_t ny, size_t nx, size_t offset)
{

	cudaMemcpy(to_d, &from_d[(nz-1)*nx*ny - offset*nx*ny],
			   nx*ny*sizeof(T), cudaMemcpyDeviceToDevice);

}


template<typename T>
void set_top(T* from_d, T* to_d,
			 size_t nz, size_t ny, size_t nx)
{

	cudaMemcpy(&to_d[(nz-1)*nx*ny], from_d,
			   nx*ny*sizeof(T), cudaMemcpyDeviceToDevice);

}


template<typename T>
void extract_bottom(T* from_d, T* to_d,
					size_t nz, size_t ny, size_t nx, size_t offset)
{

	cudaMemcpy(to_d, &from_d[offset*nx*ny], nx*ny*sizeof(T), cudaMemcpyDeviceToDevice);

}


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


template<typename T>
void extract_west(T* from, T* to,
				  size_t nz, size_t ny, size_t nx, size_t offset)
{

	dim3 blocks(32, 32);
	dim3 threads(16, 16, 1);
	extract_west_kernel<<<blocks, threads>>>(from, to, nz, ny, nx, offset);

}


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


template<typename T>
void extract_east(T* from, T* to,
				  size_t nz, size_t ny, size_t nx, size_t offset)
{

	dim3 blocks(32, 32);
	dim3 threads(16, 16, 1);
	extract_east_kernel<<<blocks, threads>>>(from, to, nz, ny, nx, offset);

}


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


template<typename T>
void extract_north(T* from, T* to,
				   size_t nz, size_t ny, size_t nx, size_t offset)
{

	dim3 blocks(32, 32);
	dim3 threads(16, 16, 1);
	extract_north_kernel<<<blocks, threads>>>(from, to, nz, ny, nx, offset);

}


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


template<typename T>
void extract_south(T* from, T* to,
				   size_t nz, size_t ny, size_t nx, size_t offset)
{

	dim3 blocks(32, 32);
	dim3 threads(16, 16, 1);
	extract_south_kernel<<<blocks, threads>>>(from, to, nz, ny, nx, offset);

}


template<typename T>
void set_south(T* from, T* to,
			   size_t nz, size_t ny, size_t nx)
{

	dim3 blocks(32, 32);
	dim3 threads(16, 16, 1);
	set_south_kernel<<<blocks, threads>>>(from, to, nz, ny, nx);

}


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


template<typename T>
void set_all_boundaries( DeviceBoundarySet<T>& bs, T* to_d,
						size_t nz, size_t ny, size_t nx)
{

	set_top<T>(bs.get_top_ptr(), to_d, nz, ny, nx);

	set_bottom<T>(bs.get_bottom_ptr(), to_d, nz, ny, nx);

	set_north<T>(bs.get_north_ptr(), to_d, nz, ny, nx);
	
	set_south<T>(bs.get_south_ptr(), to_d, nz, ny, nx);

	set_west<T>(bs.get_west_ptr(), to_d, nz, ny, nx);

	set_east<T>(bs.get_east_ptr(), to_d, nz, ny, nx);
	
}


template<typename T>
void extract_all_boundaries(T* from_d, DeviceBoundarySet<T>& bs,
							size_t nz, size_t ny, size_t nx)
{

	extract_top<T>(from_d, bs.get_top_ptr(), nz, ny, nx);

	extract_bottom<T>(from_d, bs.get_bottom_ptr(), nz, ny, nx);

	extract_north<T>(from_d, bs.get_north_ptr(), nz, ny, nx);
	
	extract_south<T>(from_d, bs.get_south_ptr(), nz, ny, nx);

	extract_west<T>(from_d, bs.get_west_ptr(), nz, ny, nx);

	extract_east<T>(from_d, bs.get_east_ptr(), nz, ny, nx);

}


template<typename T>
void extract_all_boundaries(T* from_d, DeviceBoundarySet<T>& bs,
							size_t nz, size_t ny, size_t nx, size_t offset)
{

	extract_top<T>(from_d, bs.get_top_ptr(), nz, ny, nx, offset);

	extract_bottom<T>(from_d, bs.get_bottom_ptr(), nz, ny, nx, offset);

	extract_north<T>(from_d, bs.get_north_ptr(), nz, ny, nx, offset);
	
	extract_south<T>(from_d, bs.get_south_ptr(), nz, ny, nx, offset);

	extract_west<T>(from_d, bs.get_west_ptr(), nz, ny, nx, offset);

	extract_east<T>(from_d, bs.get_east_ptr(), nz, ny, nx, offset);

}


template<typename T, class Vector>
template<class FromBoundarySet>
void BoundarySet<T, Vector>::copy(FromBoundarySet& from)
{

	thrust::copy(from.north.begin(),  from.north.end(),  north.begin());
	thrust::copy(from.south.begin(),  from.south.end(),  south.begin());
	thrust::copy(from.west.begin(),   from.west.end(),   west.begin());
	thrust::copy(from.east.begin(),   from.east.end(),   east.begin());
	thrust::copy(from.top.begin(),    from.top.end(),    top.begin());
	thrust::copy(from.bottom.begin(), from.bottom.end(), bottom.begin());
				
}

template<typename T, class Vector>
T* BoundarySet<T, Vector>::get_north_ptr()
{
	return thrust::raw_pointer_cast(&north[0]);
}

template<typename T, class Vector>
T* BoundarySet<T, Vector>::get_south_ptr()
{
	return thrust::raw_pointer_cast(&south[0]);
}

template<typename T, class Vector>
T* BoundarySet<T, Vector>::get_west_ptr()
{
	return thrust::raw_pointer_cast(&west[0]);
}

template<typename T, class Vector>
T* BoundarySet<T, Vector>::get_east_ptr()
{
	return thrust::raw_pointer_cast(&east[0]);
}

template<typename T, class Vector>
T* BoundarySet<T, Vector>::get_top_ptr()
{
	return thrust::raw_pointer_cast(&top[0]);
}

template<typename T, class Vector>
T* BoundarySet<T, Vector>::get_bottom_ptr()
{
	return thrust::raw_pointer_cast(&bottom[0]);
}

template<typename T, class Vector>
BoundarySet<T, Vector>::BoundarySet(size_t nz_, size_t ny_, size_t nx_):
	nz(nz_), ny(ny_), nx(nx_),
	size_north(nz_*nx_), size_south(nz_*nx_),
	size_west(nz_*ny_),  size_east(nz_*ny_),
	size_top(ny_*nx_),   size_bottom(ny_*nx_),
	north(nz_*nx_, 0),   south(nz_*nx_, 0),
	west(nz_*ny_, 0),    east(nz_*ny_, 0),
	top(ny_*nx_, 0),     bottom(ny_*nx_, 0) {;}

