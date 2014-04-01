/** @file grid-points.cu a test where grid points are sorted into a grid */

#include <halloc.h>

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/** a macro for checking CUDA calls */
#define cucheck(call)																										\
	{																																			\
	cudaError_t cucheck_err = (call);																			\
	if(cucheck_err != cudaSuccess) {																			\
		const char* err_str = cudaGetErrorString(cucheck_err);							\
		fprintf(stderr, "%s (%d): %s in %s\n", __FILE__, __LINE__, err_str, #call);	\
		exit(-1);																														\
	}																																			\
	}

int divup(int a, int b) { return a / b + (a % b ? 1 : 0); }

/** a simple 3d vector */
template<class T>
struct vec3 {
	T x, y, z;
	__host__ __device__ vec3(T x, T y, T z) {
		this->x = x; this->y = y; this->z = z;
	}
	__host__ __device__ vec3(T r = 0) {
		this->x = this->y = this->z = r;
	}
};

typedef vec3<int> ivec3;
typedef vec3<float> fvec3;

/** a single point list */
struct point_list_t {
	/** point index */
	int ip;
	/** next element in the list, or 0 if end */
	point_list_t *next;
};

/** gets a random float value between 0 and 1 */
float frandom(void) {	
	const int rand_max = 65536;
	return (double)(random() % rand_max) / rand_max;
}

/** gets a random point within [0, 1]^3 cube */
fvec3 random_point(void) {
	return fvec3(frandom(), frandom(), frandom());
}  // random_point

typedef unsigned long long int uint64;

/** atomicCAS wrapper for pointers (arguments same as standard atomicCAS()) */
__device__ void *atomicCAS(void **address, void *compare, void *val) {
	return (void *)atomicCAS((uint64 *)address, (uint64)compare, (uint64)val);
}  // atomicCAS

/** atomicExch wrapper for void **/
__device__ void *atomicExch(void **address, void *val) {
	return (void *)atomicExch((uint64 *)address, (uint64)val);
}

/** a function to insert a point into a grid on device; this function can be
		called concurrently by multiple threads */
__device__ void insert_point
(point_list_t **grid, int ncells, const fvec3 * __restrict__ ps, int ip,
		point_list_t *plist) {
	// compute the cell
	fvec3 p = ps[ip];
	ivec3 cell;
	cell.x = max(min((int)floorf(p.x * ncells), ncells - 1), 0);
	cell.y = max(min((int)floorf(p.y * ncells), ncells - 1), 0);
	cell.z = max(min((int)floorf(p.z * ncells), ncells - 1), 0);

	// get the cell pointer
	point_list_t * volatile *pcell = grid + (cell.x + ncells * (cell.y + ncells *
																															cell.z));
	// try to take over the new start
	// TODO: add __threadfence() somewhere
	point_list_t *old = (point_list_t *)atomicExch((void **)pcell, plist);
	plist->ip = ip;
	plist->next = old;
}  // insert_point

/** frees the grid cell; one cell can be simultaneously freed by one thread only
		*/
__device__ void free_cell(point_list_t **grid, int ncells, ivec3 cell,
point_list_t *pre_chains) {
	point_list_t **pcell = grid + cell.x + ncells * (cell.y + ncells * cell.z);
	// free all cells
	point_list_t *plist = *pcell, *pnext;
	while(plist) {
		pnext = plist->next;
		if(!pre_chains) {
			hafree(plist);
		}
		plist = pnext;
	}
}  // free_cell

/** the kernel to insert points into the grid */
__global__ void sort_points_k
(point_list_t **grid, int ncells, const fvec3 * __restrict__ ps,
 point_list_t *pre_chains, int n) {
	int ip = threadIdx.x + blockIdx.x * blockDim.x;
	if(ip >= n)
		return;

	// allocate memory for list element
	point_list_t *plist;
	if(pre_chains)
		plist = pre_chains + ip;
	else {
		plist = (point_list_t *)hamalloc(sizeof(point_list_t));
		//plist = new point_list_t();
	}
	if(!plist) {
		//printf("cannot allocate memory\n");
		return;
	}

	insert_point(grid, ncells, ps, ip, plist);
}  // sort_points_k

/** the kernel to free the entire grid; this is 1d kernel */
__global__ void free_grid_k
(point_list_t **grid, int ncells, point_list_t *pre_chains) {
	int ncells3 = ncells * ncells * ncells;
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= ncells3)
		return;
	ivec3 cell;
	cell.x = i % ncells;
	cell.y = i % (ncells * ncells) / ncells;
	cell.z = i / (ncells * ncells);
	free_cell(grid, ncells, cell, pre_chains);
}  // free_grid_k

// a test to fill in the grid and then free it
void grid_test(int n, int ncells, bool alloc, bool print) {
	// points
	size_t sz = n * sizeof(fvec3);
	fvec3 *ps, *d_ps;
	ps = (fvec3 *)malloc(sz);
	cucheck(cudaMalloc((void **)&d_ps, sz));
	for(int ip = 0; ip < n; ip++) {
		ps[ip] = random_point();
		//printf("point = (%lf, %lf %lf)\n", (double)ps[ip].x, (double)ps[ip].y, 
		//			 (double)ps[ip].z);
	}
	cucheck(cudaMemcpy(d_ps, ps, sz, cudaMemcpyHostToDevice));

	// grid
	int ncells3 = ncells * ncells * ncells;
	size_t grid_sz = ncells3 * sizeof(point_list_t *);
	point_list_t **d_grid;
	cucheck(cudaMalloc((void **)&d_grid, grid_sz));
	cucheck(cudaMemset(d_grid, 0, grid_sz));
	
	// pre-allocated per-point chains
	point_list_t *pre_chains = 0;
	if(!alloc) {
		cucheck(cudaMalloc((void **)&pre_chains, n * sizeof(point_list_t)));
		cucheck(cudaMemset(pre_chains, 0, n * sizeof(point_list_t)));
	}

	// fill the grid
	double t1 = omp_get_wtime();
	int bs = 128;
	sort_points_k<<<divup(n, bs), bs>>>(d_grid, ncells, d_ps, pre_chains, n);
	cucheck(cudaGetLastError());
	cucheck(cudaStreamSynchronize(0));
	double t2 = omp_get_wtime();

	// free the grid
	free_grid_k<<<divup(ncells3, bs), bs>>>(d_grid, ncells, pre_chains);
	cucheck(cudaGetLastError());
	cucheck(cudaStreamSynchronize(0));
	double t3 = omp_get_wtime();

	// free everything
	//free(ps);
	cucheck(cudaFree(d_grid));
	cucheck(cudaFree(d_ps));
	cucheck(cudaFree(pre_chains));

	// print time
	if(print) {
		printf("allocation time %.2lf ms\n", (t2 - t1) * 1e3);
		printf("free time %.2lf ms\n", (t3 - t2) * 1e3);
		printf("allocation performance %.2lf Mpoints/s\n", n / (t2 - t1) * 1e-6); 
		printf("free performance %.2lf Mpoints/s\n", n / (t3 - t2) * 1e-6);
	}  // if(print)

}  // grid_test

int main(int argc, char **argv) {
	srandom((int)time(0));
	size_t memory = 512 * 1024 * 1024;
	bool alloc = true;
	//cucheck(cudaSetDevice(0));
	ha_init(halloc_opts_t(memory));
	// warm-up run
	grid_test(10000, 8, alloc, false);
	// main run
	grid_test(1000000, 32, alloc, true);
	ha_shutdown();
}  // main
