/*
 * Test two domain convergence
 */

#include <vector>
#include <cstddef>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <mpi.h>

#include "cusp_poisson.hpp"
#include "solvers.hpp"
#include "test_utils.hpp"
#include "grid.hpp"
#include "utils.hpp"


typedef cusp::array1d<float, cusp::host_memory> HostVector;
typedef cusp::array1d<float, cusp::device_memory> DeviceVector;

void make_reference_solution(HostVector& x_full_h,
							 int nz, float dz, int ny, float dy, int nx, float dx)
{

	// Reference solution	
	HostVector   b_h(nz*ny*nx, 0);
	DeviceVector x_full(nz*ny*nx, 0);

	for(int k=0; k<nz; k++)
		for(int i=0; i<ny; i++)
			for(int j=0; j<nx; j++)
				b_h[j+i*nx+k*nx*ny] = sin(j*dx)*sin(i*dy)*sin(k*dz);

	DeviceVector b_full(b_h);
	PoissonSolver3DCUSP<float> solver_full(b_full, nz, dz, ny, dy, nx, dx);

	solver_full.solve(x_full);

	x_full_h = HostVector(x_full);

}

void build_b(std::size_t global_nz, float dz,
			 std::size_t global_ny, float dy,
			 std::size_t global_nx, float dx,
			 std::size_t overlap,
			 const std::vector<int>& grid_dim,
			 const std::vector<int>& grid_coords,
			 DeviceVector& b,
			 std::size_t& nz, std::size_t& ny, std::size_t& nx)
{

	std::size_t x_location = grid_coords[2];
	std::size_t y_location = grid_coords[1];
	std::size_t z_location = grid_coords[0];

	std::vector<std::size_t> start_vec;
	std::vector<std::size_t> end_vec;

	partition_domain(start_vec, end_vec, global_nx, grid_dim[2], overlap);
	std::size_t x_start = start_vec[x_location];
	std::size_t x_end   = end_vec[x_location];
	nx                  = x_end - x_start;

	partition_domain(start_vec, end_vec, global_ny, grid_dim[1], overlap);
	std::size_t y_start = start_vec[y_location];
	std::size_t y_end   = end_vec[y_location];
	ny                  = y_end - y_start;

	partition_domain(start_vec, end_vec, global_nz, grid_dim[0], overlap);
	std::size_t z_start = start_vec[z_location];
	std::size_t z_end   = end_vec[z_location];
	nz                  = z_end - z_start;	

	// Local domain
	HostVector   b_h(nx*ny*nz, 0);
	
	std::size_t ind;
	for(std::size_t k=z_start; k<z_end; k++)
		for(std::size_t i=y_start; i<y_end; i++)
			for(std::size_t j=x_start; j<x_end; j++)
			{
				ind = (j-x_start)+(i-y_start)*nx+(k-z_start)*nx*ny;
				b_h[ind] = sin(j*dx)*sin(i*dy)*sin(k*dz);
			}

	b = DeviceVector(b_h);

}

void build_ref_b(std::size_t global_nz, float dz,
				 std::size_t global_ny, float dy,
				 std::size_t global_nx, float dx,
				 std::size_t overlap,
				 const std::vector<int>& grid_dim,
				 const std::vector<int>& grid_coords,
				 DeviceVector& x)
{

	std::size_t x_location = grid_coords[2];
	std::size_t y_location = grid_coords[1];
	std::size_t z_location = grid_coords[0];

	std::vector<std::size_t> start_vec;
	std::vector<std::size_t> end_vec;

	partition_domain(start_vec, end_vec, global_nx, grid_dim[2], overlap);
	std::size_t x_start = start_vec[x_location];
	std::size_t x_end   = end_vec[x_location];
	std::size_t nx      = x_end - x_start;

	partition_domain(start_vec, end_vec, global_ny, grid_dim[1], overlap);
	std::size_t y_start = start_vec[y_location];
	std::size_t y_end   = end_vec[y_location];
	std::size_t ny      = y_end - y_start;

	partition_domain(start_vec, end_vec, global_nz, grid_dim[0], overlap);
	std::size_t z_start = start_vec[z_location];
	std::size_t z_end   = end_vec[z_location];
	std::size_t nz      = z_end - z_start;	

	HostVector xr;
	make_reference_solution(xr, global_nz, dz, global_ny, dy, global_nx, dx);

	// Local domain
	HostVector x_h(nx*ny*nz, 0);
	
	std::size_t ind;
	for(std::size_t k=z_start; k<z_end; k++)
		for(std::size_t i=y_start; i<y_end; i++)
			for(std::size_t j=x_start; j<x_end; j++)
			{
				ind = (j-x_start)+(i-y_start)*nx+(k-z_start)*nx*ny;
				x_h[ind] = xr[j+i*global_nx+k*global_nx*global_ny];
			}

	x = DeviceVector(x_h);

}


void poisson3d(MPI_Comm cart_comm,
			   std::vector<int> grid_dim,
			   DeviceVector& x,
			   DeviceVector& b,
			   std::size_t nz, float dz,
			   std::size_t ny, float dy,
			   std::size_t nx, float dx,
			   std::size_t overlap)
{


	int grid_rank;
	std::vector<int> grid_coords(3);
	MPI_Comm_rank(cart_comm, &grid_rank);
	MPI_Cart_coords(cart_comm, grid_rank, 3, &grid_coords[0]);

	bool has_east   = grid_coords[2] < grid_dim[2]-1;
	bool has_west   = grid_coords[2] > 0;
	bool has_north  = grid_coords[1] < grid_dim[1]-1;
	bool has_south  = grid_coords[1] > 0;
	bool has_top    = grid_coords[0] < grid_dim[0]-1;
	bool has_bottom = grid_coords[0] > 0;

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


	PoissonSolver3DCUSP<float> solver(b, nz, dz, ny, dy, nx, dx);

	DeviceBoundarySet<float> device_bs(nz, ny, nx);
	HostBoundarySet<float> host_bs(nz, ny, nx);
	HostBoundarySet<float> host_bs_r(nz, ny, nx);

	thrust::device_vector<float> tmp(nx*ny*nz, 0);


	const int n_iter = 20;
	for(int i=0; i<n_iter; i++)
	{


	    solver.solve(x);

		extract_all_boundaries(thrust::raw_pointer_cast(&x[0]), device_bs,
							   nz, ny, nx, overlap);
		host_bs.copy(device_bs);

		//cudaDeviceSynchronize();

		MPI_Request send_west, send_east, send_north, send_south, send_top, send_bottom;
		MPI_Request recv_west, recv_east, recv_north, recv_south, recv_top, recv_bottom;
		MPI_Status  wait_west, wait_east, wait_north, wait_south, wait_top, wait_bottom;

		if(has_east)
		{
			MPI_Isend(host_bs.get_east_ptr(), ny*nz, MPI_FLOAT, east, 0, cart_comm, &send_east);
			MPI_Irecv(host_bs_r.get_east_ptr(), ny*nz, MPI_FLOAT, east, 0, cart_comm, &recv_east);
		}
		
		if(has_west)
		{
			MPI_Isend(host_bs.get_west_ptr(), ny*nz, MPI_FLOAT, west, 0, cart_comm, &send_west);
			MPI_Irecv(host_bs_r.get_west_ptr(), ny*nz, MPI_FLOAT, west, 0, cart_comm, &recv_west);
		}

		if(has_north)
		{
			MPI_Isend(host_bs.get_north_ptr(), nx*nz, MPI_FLOAT, north, 0, cart_comm, &send_north);
			MPI_Irecv(host_bs_r.get_north_ptr(), nx*nz, MPI_FLOAT, north, 0, cart_comm, &recv_north);
		}

		if(has_south)
		{
			MPI_Isend(host_bs.get_south_ptr(), nx*nz, MPI_FLOAT, south, 0, cart_comm, &send_south);
			MPI_Irecv(host_bs_r.get_south_ptr(), nx*nz, MPI_FLOAT, south, 0, cart_comm, &recv_south);
		}

		if(has_top)
		{
			MPI_Isend(host_bs.get_top_ptr(), nx*ny, MPI_FLOAT, top, 0, cart_comm, &send_top);
			MPI_Irecv(host_bs_r.get_top_ptr(), nx*ny, MPI_FLOAT, top, 0, cart_comm, &recv_top);
		}

		if(has_bottom)
		{
			MPI_Isend(host_bs.get_bottom_ptr(), nx*ny, MPI_FLOAT, bottom, 0, cart_comm, &send_bottom);
			MPI_Irecv(host_bs_r.get_bottom_ptr(), nx*ny, MPI_FLOAT, bottom, 0, cart_comm, &recv_bottom);
		}


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
			   
			   

int main(int argc, char* argv[])
{

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

	DeviceVector x(nx*ny*nz, 0);
	
	poisson3d(cart_comm, dimensions, x, b,
			  nz, dz, ny, dy, nx, dx, overlap);


	DeviceVector xr;
	build_ref_b(global_nz, dz, global_ny, dy,
				global_nx, dx,
				overlap, dimensions, grid_coords, xr);

	float error = 0;
	for(int i=0; i<xr.size(); i++)
		error = max(error, std::abs(xr[i] - x[i]));

	std::cout << error << std::endl;

	// if(rank==0)
	// {		
	// 	save_vector(x, "of0.txt");
	// 	save_vector(xr, "ofxr0.txt");
	// 	save_vector(b, "ofb0.txt");

	// }
	// if(rank==1)
	// {		
	// 	save_vector(x, "of1.txt");
	// 	save_vector(xr, "ofxr1.txt");
	// 	save_vector(b, "ofb1.txt");
	// }

	// std::cout << rank << ' ' << nz << ' ' << ny << ' ' << nx << std::endl;

	MPI_Finalize();

	return 0;

}

