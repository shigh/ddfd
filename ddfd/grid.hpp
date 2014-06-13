
#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>
#include <iostream>
#include <math.h>


struct GlobalGrid
{

	// Global grid dimensions
	const std::size_t nz;
	const std::size_t ny;
	const std::size_t nx;
	const std::size_t overlap;

	// Grid deltas
	const std::size_t dz;
	const std::size_t dy;
	const std::size_t dx;

};


struct LocalGrid
{

	// Local grid dimensions
	const std::size_t nz;
	const std::size_t ny;
	const std::size_t nx;
	const std::size_t overlap;

	// Grid deltas
	const std::size_t dz;
	const std::size_t dy;
	const std::size_t dx;

	// Global grid locations
	// Use int because thats what MPI expects
	int z, y, x;
	
	bool has_west, has_east, has_north, has_south,
		 has_top, has_bottom;
	int  west, east, north, south,
		 top, bottom;

	LocalGrid(MPI_Comm cart_comm,
			  std::vector<int> grid_dim,
			  GlobalGrid globalGrid);

	
};


/*!
 * Start and endpoints of partitioned domains.
 * The end point generated is the actual array end, not one past the end.
 * \param start Fills with first array index of each domain
 * \param end Fills with last arary index of each domain (use <=, not <)
 * \param N Total number of grid points
 * \param n Number of subdomains
 * \param k Overlap
 */
void partition_domain(std::vector<std::size_t>& start, std::vector<std::size_t>& end,
					  const std::size_t N, const std::size_t n, const std::size_t k);

