
#pragma once

#include <vector>
#include <cstddef>
#include <algorithm>
#include <iostream>
#include <math.h>


void partition_domain(std::vector<std::size_t>& start, std::vector<std::size_t>& end,
					  const std::size_t N, const std::size_t n, const std::size_t k)
{

	start.resize(n);
	end.resize(n);

	const std::size_t domain_size = round(((double)(N+(n-1)*(1+k)))/((double)n));

	std::size_t last_end = k;
	for(std::size_t i=0; i<n; i++)
	{
		start[i] = last_end-k;
		end[i]   = last_end-k+domain_size-1;
		last_end = end[i];
	}

	end[n-1] = N-1;

}
