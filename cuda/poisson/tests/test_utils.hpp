/*
 * Helper functions and kernels for tests.
 *
 * This does not contain any tests.
 */

#ifndef __TEST_UTILS_H
#define __TEST_UTILS_H

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/equal.h>

/*
 * For checking that a kernel can be called on a pointer.
 */
template<typename T>
__global__ void set_to_one(T* a, int N)
{

	int x = threadIdx.x + blockIdx.x*blockDim.x;

	while( x<N )
	{
		a[x] = 1;
		x += blockDim.x;
	}

}


/*
 * For checking that a kernel can be called on a pointer.
 */
template<typename T>
__global__ void set_to_constant(T* a, int N, T k)
{

	int x = threadIdx.x + blockIdx.x*blockDim.x;

	while( x<N )
	{
		a[x] = k;
		x += blockDim.x;
	}

}


/*
 * Set b[x]=1 if a[x]==k else b[x]==0
 */
template<typename T>
__global__ void all_equal_kernel(T* a, T* b, int N, T k)
{

	int x = threadIdx.x + blockIdx.x*blockDim.x;

	while( x<N )
	{
		b[x] = a[x]==k ? 1 : 0;
		x += blockDim.x;
	}

}

// These functions are a hack, but good enough for now...
template<typename T>
bool all_equal_device(T* a_d, int N, T k)
{

	thrust::device_vector<T> b_d(N, 0);
	thrust::device_vector<T> true_d(N, 1);

	all_equal_kernel<T><<<1, 16>>>(a_d, thrust::raw_pointer_cast(&b_d[0]), N, k);

	return thrust::equal(b_d.begin(), b_d.end(), true_d.begin());

}


template<typename T>
bool all_equal_host(T* a_d, int N, T k)
{

	thrust::host_vector<T> expected(N, k);
	return thrust::equal(expected.begin(), expected.end(), &a_d[0]);

}


#endif
