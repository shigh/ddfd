
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
#include <thrust/equal.h>
#include "cusp_poisson.hpp"
#include "solvers.hpp"
#include "boundary.hpp"
#include "test_utils.hpp"

template<typename T>
struct ExtractBdyFixture
{

	const int nx;
	const int ny;
	const int nz;
	const int N;

	thrust::host_vector<T>   mat_h;
	thrust::device_vector<T> mat_d;

	thrust::host_vector<T> expected_n;
	thrust::host_vector<T> expected_s;
	thrust::host_vector<T> expected_w;
	thrust::host_vector<T> expected_e;
	thrust::host_vector<T> expected_t;
	thrust::host_vector<T> expected_b;

	DeviceBoundarySet<T> dbs;
	HostBoundarySet<T>   hbs;


	ExtractBdyFixture(int nz_, int ny_, int nx_, int offset=0):
		nz(nz_), ny(ny_), nx(nx_), N(nz_*ny_*nx_),
		mat_h(nz_*ny_*nx_), mat_d(nz_*ny_*nx_),
		dbs(nz_, ny_, nx_), hbs(nz_, ny_, nx_),
		expected_n(nx*nz), expected_s(nx*nz),
		expected_w(ny*nz), expected_e(ny*nz),
		expected_t(nx*ny), expected_b(nx*ny)
		
	{
		
		for(int i=0; i<N; i++)
			mat_h[i] = i;
		thrust::copy(mat_h.begin(), mat_h.end(), mat_d.begin());

		// North and south
		for(int z=0; z<nz; z++)
			for(int x=0; x<nx; x++)
			{
				expected_n[x+z*nx] = x + z*nx*ny + nx*(ny-1) - offset*nx;
				expected_s[x+z*nx] = x + z*nx*ny + offset*nx;
			}

		// West and east
		for(int z=0; z<nz; z++)
			for(int y=0; y<ny; y++)
			{
				expected_w[y+z*ny] = y*nx + z*nx*ny + offset;
				expected_e[y+z*ny] = y*nx + z*nx*ny + (nx-1) - offset;
			}
		
		// Top and bottom
		for(int y=0; y<ny; y++)
			for(int x=0; x<nx; x++)
			{
				expected_t[x+y*nx] = x + y*nx + nx*ny*(nz-1) - offset*nx*ny;
				expected_b[x+y*nx] = x + y*nx + offset*nx*ny;
			}
		

	}

	T* get_mat_d_ptr()
	{
		return thrust::raw_pointer_cast(&mat_d[0]);
	}

};


template<typename T>
class SetBdyFixture
{

public:

	thrust::device_vector<T> mat_d;
	thrust::device_vector<T> one_d;
	thrust::device_vector<T> ext_d;

	const int nz;
	const int ny;
	const int nx;

	SetBdyFixture(int nz_, int ny_, int nx_):
		nz(nz_), ny(ny_), nx(nx_) {;}

	T* get_mat_d_ptr()
	{
		return thrust::raw_pointer_cast(&mat_d[0]);
	}

	T* get_one_d_ptr()
	{
		return thrust::raw_pointer_cast(&one_d[0]);
	}

	T* get_ext_d_ptr()
	{
		return thrust::raw_pointer_cast(&ext_d[0]);
	}


	// Check that one_d==ext_d
	bool equiv()
	{

		return thrust::equal(one_d.begin(), one_d.end(),
							 ext_d.begin());

	}

	void rebuild(int N)
	{
		mat_d = thrust::device_vector<T>(nz*ny*nx, 0);
		one_d = thrust::device_vector<T>(N, 1);
		ext_d = thrust::device_vector<T>(N, 0);
	}
	
	void reset(Boundary bnd)
	{

		switch( bnd )
		{

		case North:
		case South:
			rebuild(nx*nz);
			break;
		case West:
		case East:
			rebuild(ny*nz);
			break;
		case Top:
		case Bottom:
			rebuild(nx*ny);
			break;

		}

	}

};


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


BOOST_AUTO_TEST_CASE( extract_bdy_small_square )
{

	const int nx = 3;
	const int ny = nx;
	const int nz = nx;

	ExtractBdyFixture<float> f(nz, ny, nx);

	extract_north<float>(f.get_mat_d_ptr(),
						 f.dbs.get_north_ptr(), nz, ny, nx);
	extract_south<float>(f.get_mat_d_ptr(),
						 f.dbs.get_south_ptr(), nz, ny, nx);
	extract_west<float>(f.get_mat_d_ptr(),
						 f.dbs.get_west_ptr(), nz, ny, nx);
	extract_east<float>(f.get_mat_d_ptr(),
						 f.dbs.get_east_ptr(), nz, ny, nx);
	extract_top<float>(f.get_mat_d_ptr(),
						 f.dbs.get_top_ptr(), nz, ny, nx);
	extract_bottom<float>(f.get_mat_d_ptr(),
						 f.dbs.get_bottom_ptr(), nz, ny, nx);
	

	f.hbs.copy(f.dbs);


	BOOST_CHECK( std::equal(f.expected_n.begin(), f.expected_n.end(),
							f.hbs.get_north_ptr()) );
	BOOST_CHECK( std::equal(f.expected_s.begin(), f.expected_s.end(),
							f.hbs.get_south_ptr()) );
	BOOST_CHECK( std::equal(f.expected_w.begin(), f.expected_w.end(),
							f.hbs.get_west_ptr()) );
	BOOST_CHECK( std::equal(f.expected_e.begin(), f.expected_e.end(),
							f.hbs.get_east_ptr()) );
	BOOST_CHECK( std::equal(f.expected_t.begin(), f.expected_t.end(),
							f.hbs.get_top_ptr()) );
	BOOST_CHECK( std::equal(f.expected_b.begin(), f.expected_b.end(),
							f.hbs.get_bottom_ptr()) );

}

BOOST_AUTO_TEST_CASE( extract_bdy_large_square )
{

	const int nx = 100;
	const int ny = nx;
	const int nz = nx;

	ExtractBdyFixture<float> f(nz, ny, nx);

	extract_north<float>(f.get_mat_d_ptr(),
						 f.dbs.get_north_ptr(), nz, ny, nx);
	extract_south<float>(f.get_mat_d_ptr(),
						 f.dbs.get_south_ptr(), nz, ny, nx);
	extract_west<float>(f.get_mat_d_ptr(),
						 f.dbs.get_west_ptr(), nz, ny, nx);
	extract_east<float>(f.get_mat_d_ptr(),
						 f.dbs.get_east_ptr(), nz, ny, nx);
	extract_top<float>(f.get_mat_d_ptr(),
						 f.dbs.get_top_ptr(), nz, ny, nx);
	extract_bottom<float>(f.get_mat_d_ptr(),
						 f.dbs.get_bottom_ptr(), nz, ny, nx);
	

	f.hbs.copy(f.dbs);


	BOOST_CHECK( std::equal(f.expected_n.begin(), f.expected_n.end(),
							f.hbs.get_north_ptr()) );
	BOOST_CHECK( std::equal(f.expected_s.begin(), f.expected_s.end(),
							f.hbs.get_south_ptr()) );
	BOOST_CHECK( std::equal(f.expected_w.begin(), f.expected_w.end(),
							f.hbs.get_west_ptr()) );
	BOOST_CHECK( std::equal(f.expected_e.begin(), f.expected_e.end(),
							f.hbs.get_east_ptr()) );
	BOOST_CHECK( std::equal(f.expected_t.begin(), f.expected_t.end(),
							f.hbs.get_top_ptr()) );
	BOOST_CHECK( std::equal(f.expected_b.begin(), f.expected_b.end(),
							f.hbs.get_bottom_ptr()) );

}


BOOST_AUTO_TEST_CASE( extract_bdy_small_nonsquare )
{

	const int nx = 3;
	const int ny = nx+1;
	const int nz = ny+1;

	ExtractBdyFixture<float> f(nz, ny, nx);

	extract_north<float>(f.get_mat_d_ptr(),
						 f.dbs.get_north_ptr(), nz, ny, nx);
	extract_south<float>(f.get_mat_d_ptr(),
						 f.dbs.get_south_ptr(), nz, ny, nx);
	extract_west<float>(f.get_mat_d_ptr(),
						 f.dbs.get_west_ptr(), nz, ny, nx);
	extract_east<float>(f.get_mat_d_ptr(),
						 f.dbs.get_east_ptr(), nz, ny, nx);
	extract_top<float>(f.get_mat_d_ptr(),
						 f.dbs.get_top_ptr(), nz, ny, nx);
	extract_bottom<float>(f.get_mat_d_ptr(),
						 f.dbs.get_bottom_ptr(), nz, ny, nx);
	

	f.hbs.copy(f.dbs);


	BOOST_CHECK( std::equal(f.expected_n.begin(), f.expected_n.end(),
							f.hbs.get_north_ptr()) );
	BOOST_CHECK( std::equal(f.expected_s.begin(), f.expected_s.end(),
							f.hbs.get_south_ptr()) );
	BOOST_CHECK( std::equal(f.expected_w.begin(), f.expected_w.end(),
							f.hbs.get_west_ptr()) );
	BOOST_CHECK( std::equal(f.expected_e.begin(), f.expected_e.end(),
							f.hbs.get_east_ptr()) );
	BOOST_CHECK( std::equal(f.expected_t.begin(), f.expected_t.end(),
							f.hbs.get_top_ptr()) );
	BOOST_CHECK( std::equal(f.expected_b.begin(), f.expected_b.end(),
							f.hbs.get_bottom_ptr()) );

}


BOOST_AUTO_TEST_CASE( extract_bdy_large_nonsquare )
{

	const int nx = 50;
	const int ny = nx*2;
	const int nz = ny*2;

	ExtractBdyFixture<float> f(nz, ny, nx);

	extract_north<float>(f.get_mat_d_ptr(),
						 f.dbs.get_north_ptr(), nz, ny, nx);
	extract_south<float>(f.get_mat_d_ptr(),
						 f.dbs.get_south_ptr(), nz, ny, nx);
	extract_west<float>(f.get_mat_d_ptr(),
						 f.dbs.get_west_ptr(), nz, ny, nx);
	extract_east<float>(f.get_mat_d_ptr(),
						 f.dbs.get_east_ptr(), nz, ny, nx);
	extract_top<float>(f.get_mat_d_ptr(),
						 f.dbs.get_top_ptr(), nz, ny, nx);
	extract_bottom<float>(f.get_mat_d_ptr(),
						 f.dbs.get_bottom_ptr(), nz, ny, nx);
	

	f.hbs.copy(f.dbs);


	BOOST_CHECK( std::equal(f.expected_n.begin(), f.expected_n.end(),
							f.hbs.get_north_ptr()) );
	BOOST_CHECK( std::equal(f.expected_s.begin(), f.expected_s.end(),
							f.hbs.get_south_ptr()) );
	BOOST_CHECK( std::equal(f.expected_w.begin(), f.expected_w.end(),
							f.hbs.get_west_ptr()) );
	BOOST_CHECK( std::equal(f.expected_e.begin(), f.expected_e.end(),
							f.hbs.get_east_ptr()) );
	BOOST_CHECK( std::equal(f.expected_t.begin(), f.expected_t.end(),
							f.hbs.get_top_ptr()) );
	BOOST_CHECK( std::equal(f.expected_b.begin(), f.expected_b.end(),
							f.hbs.get_bottom_ptr()) );

}


BOOST_AUTO_TEST_CASE( set_bdy_small_square )
{

	const int nx = 3;
	const int ny = nx;
	const int nz = nx;

	SetBdyFixture<float> f(nz, ny, nx);

	f.reset(North);
	set_north<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_north<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(South);
	set_south<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_south<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(West);
	set_west<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_west<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );
	
	f.reset(East);
	set_east<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_east<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(Top);
	set_top<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_top<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(Bottom);
	set_bottom<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_bottom<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );
	
}

BOOST_AUTO_TEST_CASE( set_bdy_small_nonsquare )
{

	const int nx = 3;
	const int ny = nx+1;
	const int nz = ny+1;

	SetBdyFixture<float> f(nz, ny, nx);

	f.reset(North);
	set_north<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_north<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(South);
	set_south<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_south<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(West);
	set_west<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_west<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );
	
	f.reset(East);
	set_east<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_east<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(Top);
	set_top<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_top<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(Bottom);
	set_bottom<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_bottom<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );
	

}


BOOST_AUTO_TEST_CASE( set_bdy_large_square )
{

	const int nx = 100;
	const int ny = nx;
	const int nz = nx;

	SetBdyFixture<float> f(nz, ny, nx);

	f.reset(North);
	set_north<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_north<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(South);
	set_south<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_south<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(West);
	set_west<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_west<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );
	
	f.reset(East);
	set_east<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_east<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(Top);
	set_top<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_top<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(Bottom);
	set_bottom<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_bottom<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );
	

}


BOOST_AUTO_TEST_CASE( set_bdy_large_nonsquare )
{

	const int nx = 50;
	const int ny = nx*2;
	const int nz = ny*2;

	SetBdyFixture<float> f(nz, ny, nx);

	f.reset(North);
	set_north<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_north<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(South);
	set_south<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_south<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(West);
	set_west<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_west<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );
	
	f.reset(East);
	set_east<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_east<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(Top);
	set_top<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_top<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );

	f.reset(Bottom);
	set_bottom<float>(f.get_one_d_ptr(), f.get_mat_d_ptr(),
					 nz, ny, nx);
	extract_bottom<float>(f.get_mat_d_ptr(), f.get_ext_d_ptr(),
						 nz, ny, nx);
	BOOST_CHECK( f.equiv() );
	

}


BOOST_AUTO_TEST_CASE( extract_bdy_small_square_off )
{

	const int nx = 3;
	const int ny = nx;
	const int nz = nx;
	const int offset = 1;

	ExtractBdyFixture<float> f(nz, ny, nx, offset);

	extract_north<float>(f.get_mat_d_ptr(),
						 f.dbs.get_north_ptr(), nz, ny, nx, offset);
	extract_south<float>(f.get_mat_d_ptr(),
						 f.dbs.get_south_ptr(), nz, ny, nx, offset);
	extract_west<float>(f.get_mat_d_ptr(),
						 f.dbs.get_west_ptr(), nz, ny, nx, offset);
	extract_east<float>(f.get_mat_d_ptr(),
						 f.dbs.get_east_ptr(), nz, ny, nx, offset);
	extract_top<float>(f.get_mat_d_ptr(),
						 f.dbs.get_top_ptr(), nz, ny, nx, offset);
	extract_bottom<float>(f.get_mat_d_ptr(),
						 f.dbs.get_bottom_ptr(), nz, ny, nx, offset);
	

	f.hbs.copy(f.dbs);


	BOOST_CHECK( std::equal(f.expected_n.begin(), f.expected_n.end(),
							f.hbs.get_north_ptr()) );
	BOOST_CHECK( std::equal(f.expected_s.begin(), f.expected_s.end(),
							f.hbs.get_south_ptr()) );
	BOOST_CHECK( std::equal(f.expected_w.begin(), f.expected_w.end(),
							f.hbs.get_west_ptr()) );
	BOOST_CHECK( std::equal(f.expected_e.begin(), f.expected_e.end(),
							f.hbs.get_east_ptr()) );
	BOOST_CHECK( std::equal(f.expected_t.begin(), f.expected_t.end(),
							f.hbs.get_top_ptr()) );
	BOOST_CHECK( std::equal(f.expected_b.begin(), f.expected_b.end(),
							f.hbs.get_bottom_ptr()) );

}

BOOST_AUTO_TEST_CASE( extract_bdy_large_square_off )
{

	const int nx = 100;
	const int ny = nx;
	const int nz = nx;
	const int offset = 5;

	ExtractBdyFixture<float> f(nz, ny, nx, offset);

	extract_north<float>(f.get_mat_d_ptr(),
						 f.dbs.get_north_ptr(), nz, ny, nx, offset);
	extract_south<float>(f.get_mat_d_ptr(),
						 f.dbs.get_south_ptr(), nz, ny, nx, offset);
	extract_west<float>(f.get_mat_d_ptr(),
						 f.dbs.get_west_ptr(), nz, ny, nx, offset);
	extract_east<float>(f.get_mat_d_ptr(),
						 f.dbs.get_east_ptr(), nz, ny, nx, offset);
	extract_top<float>(f.get_mat_d_ptr(),
						 f.dbs.get_top_ptr(), nz, ny, nx, offset);
	extract_bottom<float>(f.get_mat_d_ptr(),
						 f.dbs.get_bottom_ptr(), nz, ny, nx, offset);
	

	f.hbs.copy(f.dbs);


	BOOST_CHECK( std::equal(f.expected_n.begin(), f.expected_n.end(),
							f.hbs.get_north_ptr()) );
	BOOST_CHECK( std::equal(f.expected_s.begin(), f.expected_s.end(),
							f.hbs.get_south_ptr()) );
	BOOST_CHECK( std::equal(f.expected_w.begin(), f.expected_w.end(),
							f.hbs.get_west_ptr()) );
	BOOST_CHECK( std::equal(f.expected_e.begin(), f.expected_e.end(),
							f.hbs.get_east_ptr()) );
	BOOST_CHECK( std::equal(f.expected_t.begin(), f.expected_t.end(),
							f.hbs.get_top_ptr()) );
	BOOST_CHECK( std::equal(f.expected_b.begin(), f.expected_b.end(),
							f.hbs.get_bottom_ptr()) );

}


BOOST_AUTO_TEST_CASE( extract_bdy_small_nonsquare_off )
{

	const int nx = 3;
	const int ny = nx+1;
	const int nz = ny+1;
	const int offset = 1;

	ExtractBdyFixture<float> f(nz, ny, nx, offset);

	extract_north<float>(f.get_mat_d_ptr(),
						 f.dbs.get_north_ptr(), nz, ny, nx, offset);
	extract_south<float>(f.get_mat_d_ptr(),
						 f.dbs.get_south_ptr(), nz, ny, nx, offset);
	extract_west<float>(f.get_mat_d_ptr(),
						 f.dbs.get_west_ptr(), nz, ny, nx, offset);
	extract_east<float>(f.get_mat_d_ptr(),
						 f.dbs.get_east_ptr(), nz, ny, nx, offset);
	extract_top<float>(f.get_mat_d_ptr(),
						 f.dbs.get_top_ptr(), nz, ny, nx, offset);
	extract_bottom<float>(f.get_mat_d_ptr(),
						 f.dbs.get_bottom_ptr(), nz, ny, nx, offset);
	

	f.hbs.copy(f.dbs);


	BOOST_CHECK( std::equal(f.expected_n.begin(), f.expected_n.end(),
							f.hbs.get_north_ptr()) );
	BOOST_CHECK( std::equal(f.expected_s.begin(), f.expected_s.end(),
							f.hbs.get_south_ptr()) );
	BOOST_CHECK( std::equal(f.expected_w.begin(), f.expected_w.end(),
							f.hbs.get_west_ptr()) );
	BOOST_CHECK( std::equal(f.expected_e.begin(), f.expected_e.end(),
							f.hbs.get_east_ptr()) );
	BOOST_CHECK( std::equal(f.expected_t.begin(), f.expected_t.end(),
							f.hbs.get_top_ptr()) );
	BOOST_CHECK( std::equal(f.expected_b.begin(), f.expected_b.end(),
							f.hbs.get_bottom_ptr()) );

}


BOOST_AUTO_TEST_CASE( extract_bdy_large_nonsquare_off )
{

	const int nx = 50;
	const int ny = nx*2;
	const int nz = ny*2;
	const int offset = 5;

	ExtractBdyFixture<float> f(nz, ny, nx, offset);

	extract_north<float>(f.get_mat_d_ptr(),
						 f.dbs.get_north_ptr(), nz, ny, nx, offset);
	extract_south<float>(f.get_mat_d_ptr(),
						 f.dbs.get_south_ptr(), nz, ny, nx, offset);
	extract_west<float>(f.get_mat_d_ptr(),
						 f.dbs.get_west_ptr(), nz, ny, nx, offset);
	extract_east<float>(f.get_mat_d_ptr(),
						 f.dbs.get_east_ptr(), nz, ny, nx, offset);
	extract_top<float>(f.get_mat_d_ptr(),
						 f.dbs.get_top_ptr(), nz, ny, nx, offset);
	extract_bottom<float>(f.get_mat_d_ptr(),
						 f.dbs.get_bottom_ptr(), nz, ny, nx, offset);
	

	f.hbs.copy(f.dbs);


	BOOST_CHECK( std::equal(f.expected_n.begin(), f.expected_n.end(),
							f.hbs.get_north_ptr()) );
	BOOST_CHECK( std::equal(f.expected_s.begin(), f.expected_s.end(),
							f.hbs.get_south_ptr()) );
	BOOST_CHECK( std::equal(f.expected_w.begin(), f.expected_w.end(),
							f.hbs.get_west_ptr()) );
	BOOST_CHECK( std::equal(f.expected_e.begin(), f.expected_e.end(),
							f.hbs.get_east_ptr()) );
	BOOST_CHECK( std::equal(f.expected_t.begin(), f.expected_t.end(),
							f.hbs.get_top_ptr()) );
	BOOST_CHECK( std::equal(f.expected_b.begin(), f.expected_b.end(),
							f.hbs.get_bottom_ptr()) );

}


BOOST_AUTO_TEST_SUITE_END()
