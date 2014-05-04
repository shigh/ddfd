
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


template<typename T>
T* BoundarySet<T>::get_north_ptr()
{
	return north;
}

template<typename T>
T* BoundarySet<T>::get_south_ptr()
{
	return south;
}

template<typename T>
T* BoundarySet<T>::get_west_ptr()
{
	return west;
}

template<typename T>
T* BoundarySet<T>::get_east_ptr()
{
	return east;
}

template<typename T>
T* BoundarySet<T>::get_top_ptr()
{
	return top;
}

template<typename T>
T* BoundarySet<T>::get_bottom_ptr()
{
	return bottom;
}

template<typename T>
BoundarySet<T>::BoundarySet(size_t nz_, size_t ny_, size_t nx_, cudaMemoryType memory_space_):
	nz(nz_), ny(ny_), nx(nx_),
	memory_space(memory_space_),
	size_north(nz_*nx_), size_south(nz_*nx_),
	size_west(nz_*ny_),  size_east(nz_*ny_),
	size_top(ny_*nx_),   size_bottom(ny_*nx_)
{
	//init_all_buffers();
}


template<typename T>
BoundarySet<T>::~BoundarySet()
{
	//free_all_buffers();
}


template<typename T>
template<class BS>
void BoundarySet<T>::copy(BS& from)
{

	cudaMemcpyKind kind;	
	if(from.memory_space  == cudaMemoryTypeDevice &&
	   memory_space == cudaMemoryTypeHost)
		kind = cudaMemcpyDeviceToHost;
	if(from.memory_space  == cudaMemoryTypeHost &&
	   memory_space == cudaMemoryTypeDevice)
		kind = cudaMemcpyHostToDevice;
	if(from.memory_space  == cudaMemoryTypeHost &&
	   memory_space == cudaMemoryTypeHost)
		kind = cudaMemcpyHostToHost;
	if(from.memory_space  == cudaMemoryTypeDevice &&
	   memory_space == cudaMemoryTypeDevice)
		kind = cudaMemcpyDeviceToDevice;

	cudaMemcpy(north, from.get_north_ptr(),
			   size_north*sizeof(T), kind);
	cudaMemcpy(south, from.get_south_ptr(),
			   size_south*sizeof(T), kind);
	cudaMemcpy(west, from.get_west_ptr(),
			   size_west*sizeof(T), kind);
	cudaMemcpy(east, from.get_east_ptr(),
			   size_east*sizeof(T), kind);
	cudaMemcpy(top, from.get_top_ptr(),
			   size_top*sizeof(T), kind);
	cudaMemcpy(bottom, from.get_bottom_ptr(),
			   size_bottom*sizeof(T), kind);

}


template<typename T>
void BoundarySet<T>::init_all_buffers()
{

	allocate_buffer(north,  size_north);
	allocate_buffer(south,  size_south);
	allocate_buffer(west,   size_west);
	allocate_buffer(east,   size_east);
	allocate_buffer(top,    size_top);
	allocate_buffer(bottom, size_bottom);

}


template<typename T>
void BoundarySet<T>::free_all_buffers()
{

	free_buffer(north);
	free_buffer(south);
	free_buffer(west);
	free_buffer(east);
	free_buffer(top);
	free_buffer(bottom);

}


template<typename T>
void DeviceBoundarySet<T>::allocate_buffer(T*& buf, size_t N)
{

	cudaMalloc((void**)&buf, sizeof(T)*N);

}


template<typename T>
void DeviceBoundarySet<T>::free_buffer(T*& buf)
{

	cudaFree((void*)buf);

}


template<typename T>
void HostBoundarySet<T>::allocate_buffer(T*& buf, size_t N)
{

	buf = (T*)malloc(sizeof(T)*N);

}


template<typename T>
void HostBoundarySet<T>::free_buffer(T*& buf)
{

	free(buf);

}
