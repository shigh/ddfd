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


void make_reference_solution(cusp::array1d<float, cusp::host_memory>& x_full_h,
							 int nz, float dz, int ny, float dy, int nx, float dx)
{

	// Reference solution	
	cusp::array1d<float, cusp::host_memory>   b_h(nz*ny*nx, 0);
	cusp::array1d<float, cusp::device_memory> x_full(nz*ny*nx, 0);

	for(int k=0; k<nz; k++)
		for(int i=0; i<ny; i++)
			for(int j=0; j<nx; j++)
				b_h[j+i*nx+k*nx*ny] = sin(j*dx)*sin(i*dy)*sin(k*dz);

	cusp::array1d<float, cusp::device_memory> b_full(b_h);
	PoissonSolver3DCUSP<float> solver_full(b_full, nz, dz, ny, dy, nx, dx);

	solver_full.solve(x_full);

	x_full_h = cusp::array1d<float, cusp::host_memory>(x_full);

}

void build_b(std::size_t global_nz, float dz,
			 std::size_t global_ny, float dy,
			 std::size_t global_nx, float dx,
			 std::size_t overlap,
			 const std::vector<int>& grid_coords,
			 cusp::array1d<float, cusp::device_memory>& b,
			 std::size_t& nz, std::size_t& ny, std::size_t& nx)
{

	std::vector<std::size_t> start_vec;
	std::vector<std::size_t> end_vec;
	partition_domain(start_vec, end_vec, global_nz, 2, overlap);

	std::size_t x_location = grid_coords[2];
	std::size_t y_location = grid_coords[1];
	std::size_t z_location = grid_coords[0];

	std::size_t x_start = start_vec[x_location];
	std::size_t x_end   = end_vec[x_location];
	nx                  = x_end - x_start;
	std::size_t y_start = 0;//start_vec[y_location];
	std::size_t y_end   = global_ny;//end_vec[y_location];
	ny                  = y_end - y_start;
	std::size_t z_start = 0;//start_vec[z_location];
	std::size_t z_end   = global_nz;//end_vec[z_location];
	nz                  = z_end - z_start;	

	// Local domain
	cusp::array1d<float, cusp::host_memory>   b_h(nx*ny*nz, 0);
	
	std::size_t ind;
	for(std::size_t k=z_start; k<z_end; k++)
		for(std::size_t i=y_start; i<y_end; i++)
			for(std::size_t j=x_start; j<x_end; j++)
			{
				ind = (j-x_start)+(i-y_start)*nx+(k-z_start)*nx*ny;
				b_h[ind] = sin(j*dx)*sin(i*dy)*sin(k*dz);
			}

	b = cusp::array1d<float, cusp::device_memory>(b_h);

}

void build_ref_b(std::size_t global_nz, float dz,
				 std::size_t global_ny, float dy,
				 std::size_t global_nx, float dx,
				 std::size_t overlap,
				 const std::vector<int>& grid_coords,
				 cusp::array1d<float, cusp::device_memory>& x)
{

	std::vector<std::size_t> start_vec;
	std::vector<std::size_t> end_vec;
	partition_domain(start_vec, end_vec, global_nz, 2, overlap);

	std::size_t x_location = grid_coords[2];
	std::size_t y_location = grid_coords[1];
	std::size_t z_location = grid_coords[0];

	std::size_t x_start = start_vec[x_location];
	std::size_t x_end   = end_vec[x_location];
	std::size_t nx      = x_end - x_start;
	std::size_t y_start = 0;//start_vec[y_location];
	std::size_t y_end   = global_ny;//end_vec[y_location];
	std::size_t ny      = y_end - y_start;
	std::size_t z_start = 0;//start_vec[z_location];
	std::size_t z_end   = global_nz;//end_vec[z_location];
	std::size_t nz      = z_end - z_start;	

	cusp::array1d<float, cusp::host_memory> xr;
	make_reference_solution(xr, global_nz, dz, global_ny, dy, global_nx, dx);

	// Local domain
	cusp::array1d<float, cusp::host_memory> x_h(nx*ny*nz, 0);
	
	std::size_t ind;
	for(std::size_t k=z_start; k<z_end; k++)
		for(std::size_t i=y_start; i<y_end; i++)
			for(std::size_t j=x_start; j<x_end; j++)
			{
				ind = (j-x_start)+(i-y_start)*nx+(k-z_start)*nx*ny;
				x_h[ind] = xr[j+i*nx+k*nx*ny];
			}

	x = cusp::array1d<float, cusp::device_memory>(x_h);

}


void poisson3d(MPI_Comm cart_comm,
			   std::vector<int> grid_dim,
			   cusp::array1d<float, cusp::device_memory> x,
			   cusp::array1d<float, cusp::device_memory> b,
			   std::size_t nz, float dz,
			   std::size_t ny, float dy,
			   std::size_t nx, float dx,
			   std::size_t overlap)
{


	int grid_rank;
	std::vector<int> grid_coords(3);
	MPI_Comm_rank(cart_comm, &grid_rank);
	MPI_Cart_coords(cart_comm, grid_rank, 3, &grid_coords[0]);

	bool has_east = grid_coords[2] < grid_dim[2]-1;
	bool has_west = grid_coords[2] > 0;

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


	PoissonSolver3DCUSP<float> solver(b, nz, dz, ny, dy, nx, dx);

	DeviceBoundarySet<float> device_bs(nz, ny, nx);
	HostBoundarySet<float> host_bs(nz, ny, nx);
	HostBoundarySet<float> host_bs_r(nz, ny, nx);

	thrust::device_vector<float> tmp(nx*ny*nz, 0);


	const int n_iter = 10;
	for(int i=0; i<n_iter; i++)
	{

		std::cout << i << std::endl;

	    solver.solve(x);

		extract_all_boundaries(thrust::raw_pointer_cast(&x[0]), device_bs,
							   nz, ny, nx, overlap);
		host_bs.copy(device_bs);

		cudaDeviceSynchronize();

		MPI_Status east_status, west_status;

		if(has_east)
			MPI_Send(host_bs.get_east_ptr(), ny*nz, MPI_FLOAT, east, 0, cart_comm);
		if(has_east)
			MPI_Recv(host_bs_r.get_east_ptr(), ny*nz, MPI_FLOAT, east, 0, cart_comm, &east_status);
		
		if(has_west)
			MPI_Send(host_bs.get_west_ptr(), ny*nz, MPI_FLOAT, west, 0, cart_comm);
		if(has_west)
			MPI_Recv(host_bs_r.get_west_ptr(), ny*nz, MPI_FLOAT, west, 0, cart_comm, &west_status);

		device_bs.copy(host_bs_r);		

		if(has_east)
			set_east<float>(device_bs.get_east_ptr(), thrust::raw_pointer_cast(&x[0]),
							nz, ny, nx);

		if(has_west)
			set_west<float>(device_bs.get_west_ptr(), thrust::raw_pointer_cast(&x[0]),
							nz, ny, nx);


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
	dimensions[0] = 1; dimensions[1] = 1; dimensions[2] = size;

	MPI_Comm cart_comm;
	MPI_Cart_create(MPI_COMM_WORLD, 3, &dimensions[0],
					&wrap_around[0], 1, &cart_comm);


	int grid_rank;
	std::vector<int> grid_coords(3);
	MPI_Comm_rank(cart_comm, &grid_rank);
	MPI_Cart_coords(cart_comm, grid_rank, 3, &grid_coords[0]);


	std::size_t global_nx = 10;
	std::size_t global_ny = global_nx;
	std::size_t global_nz = global_ny;

	float dx = 2*M_PI/(global_nx-1.);
	float dy = 2*M_PI/(global_ny-1.);
	float dz = 2*M_PI/(global_nz-1.);

	std::size_t overlap = 2;

	std::size_t nz, ny, nx;
	cusp::array1d<float, cusp::device_memory> b;
	build_b(global_nz, dz, global_ny, dy, global_nx, dx,
			overlap, grid_coords, b,
			nz, ny, nx);

	cusp::array1d<float, cusp::device_memory> x(nx*ny*nz, 0);
	
	poisson3d(cart_comm, dimensions, x, b,
			  nz, dz, ny, dy, nx, dx, overlap);


	cusp::array1d<float, cusp::device_memory> xr;
	build_ref_b(global_nz, dz, global_ny, dy,
				global_nx, dx,
				overlap, grid_coords, xr);

	float error = 0;
	for(int i=0; i<xr.size(); i++)
		error = max(error, (xr[i] - x[i])*(xr[i] - x[i]));

	std::cout << error << std::endl;

	if(rank==0)
	{		
		save_vector(x, "of.txt");
		save_vector(xr, "ofb.txt");
	}

	MPI_Finalize();

	return 0;

}

