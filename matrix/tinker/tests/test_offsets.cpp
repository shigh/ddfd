#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <vector>
#include <algorithm>
#include <iostream>

#include "../offsets.hpp"

BOOST_AUTO_TEST_SUITE( offset_tests )

/*
 * Make sure Offsets compiles with different sets
 * of likely parameters
 */
BOOST_AUTO_TEST_CASE( template_params )
{

  Offsets<1> off1;
  Offsets<2> off2;
  Offsets<3> off3;

  Offsets<1, int[1]> off7;
  Offsets<2, int[2]> off8;
  Offsets<3, int[3]> off9;

}

BOOST_AUTO_TEST_CASE( constructors )
{

  Offsets<3> off1;
  BOOST_CHECK( off1.get_dim(0) == 0 &&
	       off1.get_dim(1) == 0 &&
	       off1.get_dim(2) == 0 );

  std::vector<int> dims = {1, 2, 3};
  Offsets<3> off2(dims);
  BOOST_CHECK( off2.get_dim(0) == 1 &&
	       off2.get_dim(1) == 2 &&
	       off2.get_dim(2) == 3 );

  
}


BOOST_AUTO_TEST_CASE( offsets_1d )
{

  std::array<int, 1> dims = {5};
  Offsets<1> off(dims);

  int expected = 3;
  int ind = off[3];

  BOOST_CHECK_EQUAL( ind, expected );
  
}


BOOST_AUTO_TEST_CASE( offsets_2d )
{

  std::array<int, 2> dims = {3, 4};
  Offsets<2> off(dims);

  int expected = 1*(4) + 2;
  int ind = off[1][2];

  BOOST_CHECK_EQUAL( ind, expected );
  
}


BOOST_AUTO_TEST_CASE( offsets_3d )
{

  std::array<int, 3> dims = {3, 4, 5};
  Offsets<3> off(dims);

  int expected = 1*(4*5) + 2*(5) + 3;
  int ind = off[1][2][3];

  BOOST_CHECK_EQUAL( ind, expected );
  
}

BOOST_AUTO_TEST_CASE( offsets_1d_c )
{

  std::array<int, 1> dims = {5};
  Offsets<1, int[1]> off(dims);

  int expected = 3;
  int ind = off[3];

  BOOST_CHECK_EQUAL( ind, expected );
  
}


BOOST_AUTO_TEST_CASE( offsets_2d_c )
{

  std::array<int, 2> dims = {3, 4};
  Offsets<2, int[2]> off(dims);

  int expected = 1*(4) + 2;
  int ind = off[1][2];

  BOOST_CHECK_EQUAL( ind, expected );
  
}


BOOST_AUTO_TEST_CASE( offsets_3d_c )
{

  std::array<int, 3> dims = {3, 4, 5};
  Offsets<3, int[3]> off(dims);

  int expected = 1*(4*5) + 2*(5) + 3;
  int ind = off[1][2][3];

  BOOST_CHECK_EQUAL( ind, expected );
  
}


BOOST_AUTO_TEST_SUITE_END()
