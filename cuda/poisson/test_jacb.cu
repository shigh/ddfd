#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE jacbtests
#include <boost/test/unit_test.hpp>
#include "utils.hpp"
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

BOOST_AUTO_TEST_SUITE( jacb_utils )

BOOST_AUTO_TEST_CASE( l_inf_device )
{

  thrust::device_vector<float> a(10, 1);
  thrust::device_vector<float> b(10, 0);
  b[2] = 10;

  float max_diff = l_inf_diff(a, b);
  BOOST_CHECK_EQUAL(max_diff, 9);

}

BOOST_AUTO_TEST_CASE( copy_bdy_device )
{

  const int nx = 3;
  const int ny = 4;
  const int N  = nx*ny;

  thrust::host_vector<float> a_h(N), b_h(N);
  thrust::device_vector<float> a_d(N, 0), b_d(N, 0);
  for(int i=0; i<a_h.size(); i++)
    {
      a_h[i] = i;
      b_h[i] = i;
    }

  thrust::copy(a_h.begin(), a_h.end(), a_d.begin());

  copy_boundaries(a_d, b_d, nx, ny);

  thrust::copy(b_d.begin(), b_d.end(), b_h.begin());

  float expected[] = {0, 1, 2,
		      3, 0, 5,
		      6, 0, 8,
		      9, 10, 11};

  BOOST_CHECK( std::equal(b_h.begin(), b_h.end(), expected) );

}

BOOST_AUTO_TEST_SUITE_END()

