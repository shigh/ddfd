
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <math.h>
#include <algorithm>
#include <cusp/coo_matrix.h>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include <cusp/krylov/cg.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "cusp_poisson.hpp"
#include "solvers.hpp"
#include "boundary.hpp"
#include "test_utils.hpp"

BOOST_AUTO_TEST_SUITE( boundary_set_tests )

// Check for explosions
BOOST_AUTO_TEST_CASE( host_constructor )
{

	int nx, ny, nz;
	nx = ny = nz = 100;
	HostBoundarySet<float> hbs(nz, ny, nx);

}


BOOST_AUTO_TEST_CASE( device_constructor )
{

	int nx, ny, nz;
	nx = ny = nz = 100;
	DeviceBoundarySet<float> hbs(nz, ny, nx);

}


BOOST_AUTO_TEST_CASE( host_get_ptr_not_null )
{

	int nx, ny, nz;
	nx = ny = nz = 100;

	HostBoundarySet<float> hbs(nz, ny, nx);

	float* north  = hbs.get_north_ptr();
	float* south  = hbs.get_south_ptr();
	float* west   = hbs.get_west_ptr();
	float* east   = hbs.get_east_ptr();
	float* top    = hbs.get_top_ptr();
	float* bottom = hbs.get_bottom_ptr();

	BOOST_CHECK( north!=NULL && south!=NULL &&
				 west!=NULL  && east!=NULL &&
				 top!=NULL   && bottom!=NULL );

}


BOOST_AUTO_TEST_CASE( device_get_ptr_not_null )
{

	int nx, ny, nz;
	nx = ny = nz = 100;

	DeviceBoundarySet<float> dbs(nz, ny, nx);

	float* north  = dbs.get_north_ptr();
	float* south  = dbs.get_south_ptr();
	float* west   = dbs.get_west_ptr();
	float* east   = dbs.get_east_ptr();
	float* top    = dbs.get_top_ptr();
	float* bottom = dbs.get_bottom_ptr();

	BOOST_CHECK( north!=NULL && south!=NULL &&
				 west!=NULL  && east!=NULL &&
				 top!=NULL   && bottom!=NULL );

}


BOOST_AUTO_TEST_CASE( device_call_kernel_on_ptrs )
{

	int nx, ny, nz;
	nx = ny = nz = 10;
	int N = nx*ny; // Same for each boundary

	DeviceBoundarySet<float> dbs(nz, ny, nx);

	float* a = dbs.get_north_ptr();
	set_to_one<float><<<1, 8>>>(a, N);
	BOOST_CHECK_EQUAL( cudaSuccess, cudaGetLastError() );

	a = dbs.get_south_ptr();
	set_to_one<float><<<1, 8>>>(a, N);
	BOOST_CHECK_EQUAL( cudaSuccess, cudaGetLastError() );

	a = dbs.get_west_ptr();
	set_to_one<float><<<1, 8>>>(a, N);
	BOOST_CHECK_EQUAL( cudaSuccess, cudaGetLastError() );

	a = dbs.get_east_ptr();
	set_to_one<float><<<1, 8>>>(a, N);
	BOOST_CHECK_EQUAL( cudaSuccess, cudaGetLastError() );

	a = dbs.get_top_ptr();
	set_to_one<float><<<1, 8>>>(a, N);
	BOOST_CHECK_EQUAL( cudaSuccess, cudaGetLastError() );

	a = dbs.get_bottom_ptr();
	set_to_one<float><<<1, 8>>>(a, N);
	BOOST_CHECK_EQUAL( cudaSuccess, cudaGetLastError() );
	
}


BOOST_AUTO_TEST_CASE( copy_host_to_device )
{

	int nx, ny, nz;
	nx = ny = nz = 10;
	int N = nx*ny; // Same size for all boundaries

	HostBoundarySet<float>   hbs(nz, ny, nx);
	DeviceBoundarySet<float> dbs(nz, ny, nx);

	std::fill_n(hbs.get_north_ptr(),  N, 1);
	std::fill_n(hbs.get_south_ptr(),  N, 2);
	std::fill_n(hbs.get_west_ptr(),   N, 3);
	std::fill_n(hbs.get_east_ptr(),   N, 4);
	std::fill_n(hbs.get_top_ptr(),    N, 5);
	std::fill_n(hbs.get_bottom_ptr(), N, 6);

	dbs.copy(hbs);

	BOOST_CHECK( all_equal_device<float>(dbs.get_north_ptr(),  N, 1) );
	BOOST_CHECK( all_equal_device<float>(dbs.get_south_ptr(),  N, 2) );
	BOOST_CHECK( all_equal_device<float>(dbs.get_west_ptr(),   N, 3) );
	BOOST_CHECK( all_equal_device<float>(dbs.get_east_ptr(),   N, 4) );
	BOOST_CHECK( all_equal_device<float>(dbs.get_top_ptr(),    N, 5) );
	BOOST_CHECK( all_equal_device<float>(dbs.get_bottom_ptr(), N, 6) );

}


BOOST_AUTO_TEST_CASE( copy_device_to_host )
{

	int nx, ny, nz;
	nx = ny = nz = 10;
	int N = nx*ny; // Same size for all boundaries

	HostBoundarySet<float>   hbs(nz, ny, nx);
	DeviceBoundarySet<float> dbs(nz, ny, nx);

	set_to_constant<float><<<1, 16>>>(dbs.get_north_ptr(),  N, 1);
	set_to_constant<float><<<1, 16>>>(dbs.get_south_ptr(),  N, 2);
	set_to_constant<float><<<1, 16>>>(dbs.get_west_ptr(),   N, 3);
	set_to_constant<float><<<1, 16>>>(dbs.get_east_ptr(),   N, 4);
	set_to_constant<float><<<1, 16>>>(dbs.get_top_ptr(),    N, 5);
	set_to_constant<float><<<1, 16>>>(dbs.get_bottom_ptr(), N, 6);

	hbs.copy(dbs);

	BOOST_CHECK( all_equal_host<float>(hbs.get_north_ptr(),  N, 1) );
	BOOST_CHECK( all_equal_host<float>(hbs.get_south_ptr(),  N, 2) );
	BOOST_CHECK( all_equal_host<float>(hbs.get_west_ptr(),   N, 3) );
	BOOST_CHECK( all_equal_host<float>(hbs.get_east_ptr(),   N, 4) );
	BOOST_CHECK( all_equal_host<float>(hbs.get_top_ptr(),    N, 5) );
	BOOST_CHECK( all_equal_host<float>(hbs.get_bottom_ptr(), N, 6) );

}


BOOST_AUTO_TEST_SUITE_END()