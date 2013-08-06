#ifndef HALLOC_GRID_H_
#define HALLOC_GRID_H_

#include "utils.h"

// constants related to grid cells
#define GRID_SIZE_LEN 6
#define GRID_ADDR_LEN 20
#define GRID_SB_LEN 13
#define GRID_INIT_POS 0
#define GRID_FIRST_SIZE_POS 1
#define GRID_SECOND_SIZE_POS 7
#define GRID_FIRST_SB_POS 13
#define GRID_SECOND_SB_POS 26
#define GRID_ADDR_POS 39
#define GRID_ADDR_SH 4
#define GRID_SB_NONE ((1 << GRID_SB_LEN) - 1)

/** initial value for the grid cell */
__host__ __device__ inline uint64 grid_cell_init() {
	uint64 no_sb_field = (1 << GRID_SB_LEN) - 1;
	return no_sb_field << GRID_FIRST_SB_POS | no_sb_field << GRID_SECOND_SB_POS;
}
/** checks whether the grid cell is initialized */
__device__ inline bool grid_is_init(uint64 cell) {
	return (cell >> GRID_INIT_POS) & 1;
} 
/** gets the first size id of the grid cell */
__device__ inline uint grid_first_size_id(uint64 cell) {
	return (cell >> GRID_FIRST_SIZE_POS) & ((1ull << GRID_SIZE_LEN) - 1);
}
/** gets the  second size id of the grid cell */
__device__ inline uint grid_second_size_id(uint64 cell) {
	return (cell >> GRID_SECOND_SIZE_POS) & ((1ull << GRID_SIZE_LEN) - 1);
}
/** gets the first superblock id of the grid cell  */
__device__ inline uint grid_first_sb_id(uint64 cell) {
	return (cell >> GRID_FIRST_SB_POS) & ((1ull << GRID_SB_LEN) - 1);
}
/** gets the second superblock id of the grid cell  */
__device__ inline uint grid_second_sb_id(uint64 cell) {
	return (cell >> GRID_SECOND_SB_POS) & ((1ull << GRID_SB_LEN) - 1);
}

#endif
