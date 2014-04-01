/** @file grid-points.cu a test where grid points are sorted into a grid */

#include <halloc.h>

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

/** sets CUDA device variable */
#define cuset(symbol, T, val)																		\
{																																\
	void *cuset_addr;																							\
	cucheck(cudaGetSymbolAddress(&cuset_addr, symbol));						\
	T cuset_val = (val);																					\
	cucheck(cudaMemcpy(cuset_addr, &cuset_val, sizeof(cuset_val), \
										 cudaMemcpyHostToDevice));									\
}  // cuset

int divup(int a, int b) { return a / b + (a % b ? 1 : 0); }

typedef unsigned long long int uint64;

/** a random value in [a, b] range */
int random(int a, int b) {
	return a + random() % (b - a + 1);
	//return a;
}

/** an array filled with random values in [a, b] range, with contiguous groups
		of p values starting at p being the same */
void random_array(int *arr, size_t n, int p, int a, int b) {
	int v = 0;
	for(size_t i = 0; i < n; i++) {
		if(i % p == 0)
			v = random(a, b);
		arr[i] = v;
	}
}

/** a list of neighboring vertices */
struct vertex_t;
struct edge_list_t {
	/** the target vertex */
	vertex_t *target;
	/** the next element in the vertex list */
	edge_list_t *next;
	/** creates a new list edge */
	__host__ __device__ edge_list_t(vertex_t *target, edge_list_t *next = 0) 
		: target(target), next(next) {}
};  // edge_list_t

/** a single vertex */
struct vertex_t {
	/** the id of the vertex (= illusion of some data) */
	int id;
	/** the number of edges in the vertex */
	int nedges;
	/** the list of edges of the vertex */
	edge_list_t *edges;
	/** create a new vertex */
	__host__ __device__ vertex_t(int id) :
		id(id), nedges(0), edges(0) {}
	/** adds an edge (to the beginning of the list) */
	__device__ void add_edge(vertex_t *target) {
		edge_list_t *new_edges = (edge_list_t *)hamalloc(sizeof(edge_list_t));
		*new_edges = edge_list_t(target, edges);
		edges = new_edges;
		nedges++;
	}  // add_edge
	/** same function on the host */
	__host__ void add_edge_host(vertex_t *target) {
		edge_list_t *new_edges = (edge_list_t *)malloc(sizeof(edge_list_t));
		*new_edges = edge_list_t(target, edges);
		edges = new_edges;
		nedges++;
	}  // add_edge

};  // vertex_t

/** random number data on device */
uint * __constant__ random_states_g;

void drandom_init(void) {
	// TODO: somehow standardize this number
	const uint MAX_NTHREADS = 8 * 1024 * 1024;
	uint n = MAX_NTHREADS;
	size_t sz = n * sizeof(uint);
	uint *d_random_states, *h_random_states;

	// allocate memory
	cucheck(cudaMalloc((void **)&d_random_states, sz));
	h_random_states = (uint *)malloc(sz);

	// initialize random values, respect groups
	//uint gp = 1;
	uint gp = 1;
	uint seed;
	for(uint i = 0; i < n; i++) {
		if(i % gp == 0)
			seed = random();
		h_random_states[i] = seed;
	}
	cucheck(cudaMemcpy(d_random_states, h_random_states, sz, 
										 cudaMemcpyHostToDevice));
	free(h_random_states);
	
	// initialize device variable
	cuset(random_states_g, uint *, d_random_states);	
}  // drandom_init

/** gets the next seed */
static inline __host__ __device__ uint next_seed(uint seed) {
	/* seed ^= (seed << 13);
	seed ^= (seed >> 17);
	seed ^= (seed << 5); */
	seed = (seed ^ 61) ^ (seed >> 16);
	seed *= 9;
	seed = seed ^ (seed >> 4);
	seed *= 0x27d4eb2d;
	seed = seed ^ (seed >> 15);
	return seed;
}  // next_seed

/** get the random value on the device */
static inline  __device__ uint drandom(void) {
	uint tid = threadIdx.x + blockIdx.x * blockDim.x;
	uint seed = random_states_g[tid];
	seed = next_seed(seed);
	random_states_g[tid] = seed;
	return seed;
}  // drandom

/** get the random value within the specified interval (both ends inclusive) on
		the device */
static inline __device__ uint drandom(uint a, uint b) {
	return a + (drandom() & 0x00ffffffu) % (uint)(b - a + 1);
}  // drandom

static inline __host__ uint hdrandom(uint *seed, uint a, uint b) {
	*seed = next_seed(*seed);
	return a + (*seed & 0x00ffffffu) % (uint)(b - a + 1);
}  // hdrandom

/** get the floating-point random value between 0 and 1 */
// static inline __device__ float drandomf(void) {
// 	float f = 1.0f / (1024.0f * 1024.0f);
// 	uint m = 1024 * 1024;
// 	return f * drandom(0, m - 1);
// }  // drandomf

/** kernel building a random graph */
__global__ void random_graph_build_k
(vertex_t *__restrict__ vs, int nvs, int max_degree) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid >= nvs)
		return;
	vertex_t v = vertex_t(tid);
	// build edges for each vertex
	int nedges = drandom(1, max_degree);
	for(int iedge = 0; iedge < nedges; iedge++) {
		vertex_t *target = &vs[drandom(1, nvs)];
		v.add_edge(target);
	}
	// write the vertex out
	vs[tid] = v;
}  // random_graph_build_k

/** random graph test on GPU */
void random_graph_gpu(int nvs, int max_degree, bool print) {
	size_t vs_sz = nvs * sizeof(vertex_t);
	vertex_t *d_vs;
	cucheck(cudaMalloc((void **)&d_vs, vs_sz));
	// build the graph
	int bs = 128;
	double t1 = omp_get_wtime();
	random_graph_build_k<<<divup(nvs, bs), bs>>>(d_vs, nvs, max_degree);
	cucheck(cudaGetLastError());
	cucheck(cudaStreamSynchronize(0));
	double t2 = omp_get_wtime();
	
	if(print) {
		double t = t2 - t1;
		double perf = 0.5 * (max_degree + 1) * nvs / t;
		printf("GPU time: %.3lf ms\n", t * 1e3);
		printf("GPU performance: %.3lf Medges/s\n", perf * 1e-6);
	}
	cucheck(cudaFree(d_vs));
}  // random_graph_gpu

/** random graph test on CPU */
void random_graph_cpu(int nvs, int max_degree, bool print) {
	size_t vs_sz = nvs * sizeof(vertex_t);
	vertex_t *vs = (vertex_t *)malloc(vs_sz);
	// build the graph
	double t1 = omp_get_wtime();
  #pragma omp parallel 
	{
		uint seed = random();
		#pragma omp for
		for(int tid = 0; tid < nvs; tid++) {
			vertex_t v = vertex_t(tid);
			// build edges for each vertex
			int nedges = hdrandom(&seed, 1, max_degree);
			for(int iedge = 0; iedge < nedges; iedge++) {
				vertex_t *target = &vs[hdrandom(&seed, 1, nvs)];
				v.add_edge_host(target);
			}
			// write the vertex out
			vs[tid] = v;
		}
  }
	double t2 = omp_get_wtime();
	
	if(print) {
		double t = t2 - t1;
		double perf = 0.5 * (max_degree + 1) * nvs / t;
		printf("CPU time: %.3lf ms\n", t * 1e3);
		printf("CPU performance: %.3lf Medges/s\n", perf * 1e-6);
	}
	free(vs);
}  // random_graph_cpu

int main(int argc, char **argv) {
	srandom((int)time(0));
	drandom_init();
	size_t memory = 512 * 1024 * 1024;
	// GPU test
	ha_init(halloc_opts_t(memory));
	//cucheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
	random_graph_gpu(10000, 4, false);
	random_graph_gpu(1000000, 8, true);
	printf("==============================\n");
	// CPU test
	random_graph_cpu(10000, 4, false);
	random_graph_cpu(1000000, 8, true);
	ha_shutdown();
}  // main
