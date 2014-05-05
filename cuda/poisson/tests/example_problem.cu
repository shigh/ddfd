
#include "example_problem.hpp"


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


void build_ref_x(std::size_t global_nz, float dz,
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

