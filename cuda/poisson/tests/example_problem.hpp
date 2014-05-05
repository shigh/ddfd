
#pragma once

#include <vector>
#include <cstddef>

#include "cusp_poisson.hpp"
#include "solvers.hpp"
#include "grid.hpp"
#include "utils.hpp"


typedef cusp::array1d<float, cusp::host_memory> HostVector;
typedef cusp::array1d<float, cusp::device_memory> DeviceVector;


/*!
 * Make full domain reference solution
 * \param[out] x_full_h Solution vector
 */
void make_reference_solution(HostVector& x_full_h,
							 int nz, float dz, int ny, float dy, int nx, float dx);

/*!
 * Build RHS
 * \param grid_dim Global grid dimensions
 * \param grid_coords Location in grid
 * \param[out] b Solution vector
 * \param[out] nz
 * \param[out] ny
 * \param[out] nx 
 */
void build_b(std::size_t global_nz, float dz,
			 std::size_t global_ny, float dy,
			 std::size_t global_nx, float dx,
			 std::size_t overlap,
			 const std::vector<int>& grid_dim,
			 const std::vector<int>& grid_coords,
			 DeviceVector& b,
			 std::size_t& nz, std::size_t& ny, std::size_t& nx);


/*!
 * Example local reference solution
 * \param grid_dim Global grid dimensions
 * \param grid_coords Location in grid
 * \param[out] x Solution vector
 */
void build_ref_x(std::size_t global_nz, float dz,
				 std::size_t global_ny, float dy,
				 std::size_t global_nx, float dx,
				 std::size_t overlap,
				 const std::vector<int>& grid_dim,
				 const std::vector<int>& grid_coords,
				 DeviceVector& x);

