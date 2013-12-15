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

  Offsets<1, int> off4;
  Offsets<2, int> off5;
  Offsets<3, int> off6;

  // Its not meant to be used with C style arrays,
  // but just in case you really want to...
  Offsets<1, int, int[1]> off7;
  Offsets<2, int, int[2]> off8;
  Offsets<3, int, int[3]> off9;
}

BOOST_AUTO_TEST_SUITE_END()
