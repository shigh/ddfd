
#include "grid.hpp"

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
		end[i]   = last_end-k+domain_size;
		last_end = last_end-k+domain_size-1;
	}

	end[n-1] = N;

}


LocalGrid::LocalGrid(MPI_Comm cart_comm,
					 std::vector<int> grid_dim,
					 GlobalGrid globalGrid) :
	nz(globalGrid.nz), ny(globalGrid.ny), nx(globalGrid.nx),
	dz(globalGrid.dz), dy(globalGrid.dy), dx(globalGrid.dx),
	overlap(globalGrid.overlap)
{

	// Get local grid location
	int grid_rank;
	std::vector<int> grid_coords(3);
	MPI_Comm_rank(cart_comm, &grid_rank);
	MPI_Cart_coords(cart_comm, grid_rank, 3, &grid_coords[0]);

	// Determine which neighboors exist
	has_east   = grid_coords[2] < grid_dim[2]-1;
	has_west   = grid_coords[2] > 0;
	has_north  = grid_coords[1] < grid_dim[1]-1;
	has_south  = grid_coords[1] > 0;
	has_top    = grid_coords[0] < grid_dim[0]-1;
	has_bottom = grid_coords[0] > 0;

	// Get neighboor ranks
	std::vector<int> tmp_coords(3);		
	east = -1;
	if(has_east)
	{
		tmp_coords = grid_coords;
		tmp_coords[2] += 1;
		MPI_Cart_rank(cart_comm, &tmp_coords[0], &east);
	}
	west = -1;
	if(has_west)
	{
		tmp_coords = grid_coords;
		tmp_coords[2] -= 1;
		MPI_Cart_rank(cart_comm, &tmp_coords[0], &west);
	}
	north = -1;
	if(has_north)
	{
		tmp_coords = grid_coords;
		tmp_coords[1] += 1;
		MPI_Cart_rank(cart_comm, &tmp_coords[0], &north);
	}
	south = -1;
	if(has_south)
	{
		tmp_coords = grid_coords;
		tmp_coords[1] -= 1;
		MPI_Cart_rank(cart_comm, &tmp_coords[0], &south);
	}
	top = -1;
	if(has_top)
	{
		tmp_coords = grid_coords;
		tmp_coords[0] += 1;
		MPI_Cart_rank(cart_comm, &tmp_coords[0], &top);
	}
	bottom = -1;
	if(has_bottom)
	{
		tmp_coords = grid_coords;
		tmp_coords[0] -= 1;
		MPI_Cart_rank(cart_comm, &tmp_coords[0], &bottom);
	}


}
