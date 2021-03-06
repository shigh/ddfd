/*
 * Test two domain convergence
 */

#include <vector>
#include <cstddef>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <mpi.h>

#include "example_problem.hpp"
#include "cusp_poisson.hpp"
#include "solvers.hpp"
#include "test_utils.hpp"
#include "grid.hpp"
#include "utils.hpp"


/*!
 * Domain decomposition for poissons equation in 3D
 * \param cart_comm Initialized 3D cart comm
 * \param grid_dim Global grid dimensions
 * \param[in,out] x Result vector and initial guess
 * \param b Right hand size
 */
void poisson3d(MPI_Comm cart_comm,
			   std::vector<int> grid_dim,
			   DeviceVector& x,
			   DeviceVector& b,
			   std::size_t nz, float dz,
			   std::size_t ny, float dy,
			   std::size_t nx, float dx,
			   std::size_t overlap)
{


	// Get local grid location
	int grid_rank;
	std::vector<int> grid_coords(3);
	MPI_Comm_rank(cart_comm, &grid_rank);
	MPI_Cart_coords(cart_comm, grid_rank, 3, &grid_coords[0]);

	// Determine which neighboors exist
	bool has_east   = grid_coords[2] < grid_dim[2]-1;
	bool has_west   = grid_coords[2] > 0;
	bool has_north  = grid_coords[1] < grid_dim[1]-1;
	bool has_south  = grid_coords[1] > 0;
	bool has_top    = grid_coords[0] < grid_dim[0]-1;
	bool has_bottom = grid_coords[0] > 0;

	// Get neighboor ranks
	std::vector<int> tmp_coords(3);		
	int east = -1;
	if(has_east)
	{
		tmp_coords = grid_coords;
		tmp_coords[2] += 1;
		MPI_Cart_rank(cart_comm, &tmp_coords[0], &east);
	}
	int west = -1;
	if(has_west)
	{
		tmp_coords = grid_coords;
		tmp_coords[2] -= 1;
		MPI_Cart_rank(cart_comm, &tmp_coords[0], &west);
	}
	int north = -1;
	if(has_north)
	{
		tmp_coords = grid_coords;
		tmp_coords[1] += 1;
		MPI_Cart_rank(cart_comm, &tmp_coords[0], &north);
	}
	int south = -1;
	if(has_south)
	{
		tmp_coords = grid_coords;
		tmp_coords[1] -= 1;
		MPI_Cart_rank(cart_comm, &tmp_coords[0], &south);
	}
	int top = -1;
	if(has_top)
	{
		tmp_coords = grid_coords;
		tmp_coords[0] += 1;
		MPI_Cart_rank(cart_comm, &tmp_coords[0], &top);
	}
	int bottom = -1;
	if(has_bottom)
	{
		tmp_coords = grid_coords;
		tmp_coords[0] -= 1;
		MPI_Cart_rank(cart_comm, &tmp_coords[0], &bottom);
	}


	// TODO Factor this into a template parameter
	// Solver setup should be in the calling code
	PoissonSolver3DCUSP<float> solver(b, nz, dz, ny, dy, nx, dx);

	DeviceBoundarySet<float> device_bs(nz, ny, nx);
	HostBoundarySet<float> host_bs(nz, ny, nx);
	HostBoundarySet<float> host_bs_r(nz, ny, nx);

	thrust::device_vector<float> tmp(nx*ny*nz, 0);


	// TODO Add as a function parameter
	const int n_iter = 20;
	for(int i=0; i<n_iter; i++)
	{


	    solver.solve(x);

		extract_all_boundaries(thrust::raw_pointer_cast(&x[0]), device_bs,
							   nz, ny, nx, overlap);
		host_bs.copy(device_bs);

		// Communicate with neighboors
		MPI_Request send_west, send_east, send_north, send_south, send_top, send_bottom;
		MPI_Request recv_west, recv_east, recv_north, recv_south, recv_top, recv_bottom;
		MPI_Status  wait_west, wait_east, wait_north, wait_south, wait_top, wait_bottom;

		// Post send and recieves
		if(has_east)
		{
			MPI_Isend(host_bs.get_east_ptr(), host_bs.size_east,
					  MPI_FLOAT, east, 0, cart_comm, &send_east);
			MPI_Irecv(host_bs_r.get_east_ptr(), host_bs.size_east,
					  MPI_FLOAT, east, 0, cart_comm, &recv_east);
		}
		
		if(has_west)
		{
			MPI_Isend(host_bs.get_west_ptr(), host_bs.size_west,
					  MPI_FLOAT, west, 0, cart_comm, &send_west);
			MPI_Irecv(host_bs_r.get_west_ptr(), host_bs.size_west,
					  MPI_FLOAT, west, 0, cart_comm, &recv_west);
		}

		if(has_north)
		{
			MPI_Isend(host_bs.get_north_ptr(), host_bs.size_north,
					  MPI_FLOAT, north, 0, cart_comm, &send_north);
			MPI_Irecv(host_bs_r.get_north_ptr(),  host_bs.size_north,
					  MPI_FLOAT, north, 0, cart_comm, &recv_north);
		}

		if(has_south)
		{
			MPI_Isend(host_bs.get_south_ptr(),  host_bs.size_south,
					  MPI_FLOAT, south, 0, cart_comm, &send_south);
			MPI_Irecv(host_bs_r.get_south_ptr(),  host_bs.size_south,
					  MPI_FLOAT, south, 0, cart_comm, &recv_south);
		}

		if(has_top)
		{
			MPI_Isend(host_bs.get_top_ptr(),  host_bs.size_top,
					  MPI_FLOAT, top, 0, cart_comm, &send_top);
			MPI_Irecv(host_bs_r.get_top_ptr(), host_bs.size_top,
					  MPI_FLOAT, top, 0, cart_comm, &recv_top);
		}

		if(has_bottom)
		{
			MPI_Isend(host_bs.get_bottom_ptr(), host_bs.size_bottom,
					  MPI_FLOAT, bottom, 0, cart_comm, &send_bottom);
			MPI_Irecv(host_bs_r.get_bottom_ptr(), host_bs.size_bottom,
					  MPI_FLOAT, bottom, 0, cart_comm, &recv_bottom);
		}

		// Wait for recvs so we can start updating solver boundaries
		if(has_west)
			MPI_Wait(&recv_west, &wait_west);
		if(has_east)
			MPI_Wait(&recv_east, &wait_east);
		if(has_north)
			MPI_Wait(&recv_north, &wait_north);
		if(has_south)
			MPI_Wait(&recv_south, &wait_south);
		if(has_top)
			MPI_Wait(&recv_top, &wait_top);
		if(has_bottom)
			MPI_Wait(&recv_bottom, &wait_south);

		// Update solver boundaries
		device_bs.copy(host_bs_r);		

		if(has_east)
			set_east<float>(device_bs.get_east_ptr(), thrust::raw_pointer_cast(&x[0]),
							nz, ny, nx);
		if(has_west)
			set_west<float>(device_bs.get_west_ptr(), thrust::raw_pointer_cast(&x[0]),
							nz, ny, nx);
		if(has_north)
			set_north<float>(device_bs.get_north_ptr(), thrust::raw_pointer_cast(&x[0]),
							nz, ny, nx);
		if(has_south)
			set_south<float>(device_bs.get_south_ptr(), thrust::raw_pointer_cast(&x[0]),
							nz, ny, nx);
		if(has_top)
			set_top<float>(device_bs.get_top_ptr(), thrust::raw_pointer_cast(&x[0]),
							nz, ny, nx);
		if(has_bottom)
			set_bottom<float>(device_bs.get_bottom_ptr(), thrust::raw_pointer_cast(&x[0]),
							nz, ny, nx);

		// Wait for all sends to complete so we can start the next iteration
		if(has_west)
			MPI_Wait(&send_west, &wait_west);
		if(has_east)
			MPI_Wait(&send_east, &wait_east);
		if(has_north)
			MPI_Wait(&send_north, &wait_north);
		if(has_south)
			MPI_Wait(&send_south, &wait_south);
		if(has_top)
			MPI_Wait(&send_top, &wait_top);
		if(has_bottom)
			MPI_Wait(&send_bottom, &wait_bottom);

	}


}
			   
			   
/*
 * Sample use of poisson3d function
 */
int main(int argc, char* argv[])
{

	// Setup MPI
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	std::vector<int> dimensions(3, 1);
	std::vector<int> wrap_around(3, 0);
	int gd = std::pow(size, 1./3);
	dimensions[0] = gd; dimensions[1] = gd; dimensions[2] = gd;

	MPI_Comm cart_comm;
	MPI_Cart_create(MPI_COMM_WORLD, 3, &dimensions[0],
					&wrap_around[0], 1, &cart_comm);


	int grid_rank;
	std::vector<int> grid_coords(3);
	MPI_Comm_rank(cart_comm, &grid_rank);
	MPI_Cart_coords(cart_comm, grid_rank, 3, &grid_coords[0]);

	// Problem setup
	std::size_t global_nx = 20;
	std::size_t global_ny = global_nx;
	std::size_t global_nz = global_ny;

	float dx = 2*M_PI/(global_nx-1.);
	float dy = 2*M_PI/(global_ny-1.);
	float dz = 2*M_PI/(global_nz-1.);

	std::size_t overlap = 2;

	std::size_t nz, ny, nx;
	DeviceVector b;
	build_b(global_nz, dz, global_ny, dy, global_nx, dx,
			overlap, dimensions, grid_coords, b,
			nz, ny, nx);

	// Solution vector
	DeviceVector x(nx*ny*nz, 0);

	// Solve
	poisson3d(cart_comm, dimensions, x, b,
			  nz, dz, ny, dy, nx, dx, overlap);


	// Get reference solution
	DeviceVector xr;
	build_ref_x(global_nz, dz, global_ny, dy,
				global_nx, dx,
				overlap, dimensions, grid_coords, xr);

	// Calc error
	float error = 0;
	for(int i=0; i<xr.size(); i++)
		error = max(error, std::abs(xr[i] - x[i]));

	std::cout << error << std::endl;

	MPI_Finalize();

	return 0;

}

