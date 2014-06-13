
#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>
#include <iostream>
#include <math.h>


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

