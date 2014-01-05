/** @file grid.cuh implementation of superblock grid */

/** base address of the grid; this is the start address of the grid. It is
		always aligned to superblock size boundary */
static void * __constant__ base_addr_g;
/** superblock grid; TODO: cache in L1, this helps */
__attribute__((aligned(128))) static __device__ uint64 sb_grid_g[2 * MAX_NSBS];

//extern __constant__ uint sb_sz_g;
//extern __constant__ uint sb_sz_sh_g;

/** add the superblock to the grid 
		// TODO: use on device as well, also with size id
*/
__host__ void grid_add_sb
(uint64 *cells, void *base_addr, uint sb, void *sb_addr, uint sb_sz) {
	void *sb_end_addr = (char *)sb_addr + sb_sz - 1;
	uint icell_start = ((char *)sb_addr - (char *)base_addr) / sb_sz;
	uint icell_end = ((char *)sb_addr + sb_sz - 1 - (char *)base_addr) / sb_sz;
	for(uint icell = icell_start; icell <= icell_end; icell++) {
		uint64 cell = cells[icell];
		cell |= 1ull << GRID_INIT_POS;
		void *cell_start_addr = (char *)base_addr + (uint64)icell * sb_sz;
		void *cell_end_addr = (char *)base_addr + (uint64)(icell + 1) * sb_sz - 1;
		if(sb_addr <= cell_start_addr) {
			// set first superblock in cell
			uint64 first_sb_mask = ((1ull << GRID_SB_LEN) - 1) << GRID_FIRST_SB_POS;
			cell = ~first_sb_mask & cell | (uint64)sb << GRID_FIRST_SB_POS;
		}
		if(sb_end_addr >= cell_end_addr) {
			// set second superblock in cell
			uint64 second_sb_mask = ((1ull << GRID_SB_LEN) - 1) << GRID_SECOND_SB_POS;
			cell = ~second_sb_mask & cell | (uint64)sb << GRID_SECOND_SB_POS;
		}
		uint64 mid_addr_mask = ((1ull << GRID_ADDR_LEN) - 1) << GRID_ADDR_POS;
		// set the break address
		if(sb_addr > cell_start_addr) {
			// current superblock is the second superblock, mid address is its start
			uint64 mid_addr = ((char *)sb_addr - (char *)cell_start_addr) >> 
				GRID_ADDR_SH;
			cell = ~mid_addr_mask & cell | mid_addr << GRID_ADDR_POS;
			//printf("icell = %d, cell_addr = %p, sb_addr = %p, mid_addr = %llx\n",
			//			 icell, cell_start_addr, sb_addr, mid_addr);
		} else if(sb_end_addr <= cell_end_addr) {
			// current superblock is the first superblock, mid address is end of this
			// superblock + 1
			uint64 mid_addr = ((char *)sb_end_addr + 1 - (char *)cell_start_addr) >>
				GRID_ADDR_SH;
			cell = ~mid_addr_mask & cell | mid_addr << GRID_ADDR_POS;
			//printf("icell = %d, cell_addr = %p, sb_addr = %p, mid_addr = %llx\n",
			//			 icell, cell_start_addr, sb_addr, mid_addr);
		}
		// save the modified cell
		cells[icell] = cell;
	}  // for(each cell in interval)
}  // grid_add_sb

/** gets the mid-address of the grid cell */
__device__ inline void *grid_mid_addr(uint icell, uint64 cell) {
	uint in_sb_addr = ((cell >> GRID_ADDR_POS) & ((1ull << GRID_ADDR_LEN) - 1))
		<< GRID_ADDR_SH;
	return (char *)base_addr_g + (uint64)icell * sb_sz_g + in_sb_addr;
}
/** gets the grid cell for the pointer */
__device__ inline uint64 grid_cell(void *p, uint *icell) {
	// TODO: handle stale cell data
	//*icell = ((char *)p - (char *)base_addr_g) / sb_sz_g;
	*icell = ((char *)p - (char *)base_addr_g) >> sb_sz_sh_g;
	//return sb_grid_g[*icell];
	return ldca(sb_grid_g + *icell);
}
/** gets the (de)allocation size id for the pointer */
__device__ inline uint grid_size_id(uint icell, uint64 cell, void *p) {
	void *midp = grid_mid_addr(icell, cell);
	return p < midp ? grid_first_size_id(cell) : grid_second_size_id(cell);
}
/** gets the (de)allocation superblock id for the pointer */
__device__ inline uint grid_sb_id(uint icell, uint64 cell, void *p) {
	//void *midp = grid_mid_addr(icell, cell);
	uint in_sb_addr = ((cell >> GRID_ADDR_POS) & ((1ull << GRID_ADDR_LEN) - 1))
		<< GRID_ADDR_SH;
	//uint in_sb_addr = ((cell >> GRID_ADDR_POS) & ((1ull << GRID_ADDR_LEN) - 1));
	//uint in_p = (char *)p - (char *)base_addr_g - ((uint64)icell << sb_sz_sh_g);
	uint in_p = (char *)p - (char *)base_addr_g - (uint64)icell * sb_sz_g;
	//uint in_p = uint(((char *)p - (char *)base_addr_g) >> GRID_ADDR_SH) - (icell <<
	//	(sb_sz_sh_g - GRID_ADDR_SH));
	//return p < midp ? grid_first_sb_id(cell) : grid_second_sb_id(cell);
	return in_p < in_sb_addr ? grid_first_sb_id(cell) : grid_second_sb_id(cell);
}
