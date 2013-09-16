/** @file statistics.cuh functions for collecting memory statistics */

/** total free memory on device, B */
__device__ uint64 free_mem_g;
/** maximum memory that can be allocated, B */
__device__ uint64 max_alloc_mem_g;
/** total Halloc memory (incl. CUDA memory), B */
__constant__ uint64 total_mem_g;
/** memory assigned to CUDA allocator, B */
__constant__ uint64 cuda_mem_g;

/** one-thread kernel determining maximum allocatable memory; it does so by
		doing binary search on what CUDA malloc can do */
__global__ void find_max_alloc_k() {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i > 0)
		return;
	uint64 hi = cuda_mem_g, lo = 0, mid;
	uint64 min_diff = 1024 * 1024;
	while(hi - lo > min_diff) {
		mid = (hi + lo) / 2;
		void *p = malloc(mid);
		if(p) {
			lo = mid;
			free(p);
		} else 
			hi = mid;
	}  // while
	max_alloc_mem_g = mid;
}  // find_max_alloc_k

/** multi-thread kernel that counts free memory available on device by launching
		one thread per slab */
__global__ void find_free_mem_k(bool ideal) {
	uint i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= nsbs_g)
		return;
	uint sb_sz = sb_sz_g;
	uint chunk_sz = sbs_g[i].chunk_sz;
	uint nused_chunks = sb_count(sb_counters_g[i]);
	uint used_mem = chunk_sz != 0 ? chunk_mul(nused_chunks, chunk_sz) : 0;
	uint free_sz = sb_sz - used_mem;
	atomicAdd(&free_mem_g, free_sz);
	if(ideal && chunk_sz == 0)
		atomicAdd(&max_alloc_mem_g, sb_sz);
	if(i == 0)
		atomicAdd(&free_mem_g, cuda_mem_g);
}  // find_free_mem_k

double ha_extfrag(bool ideal) {
	uint bs = 128;
	cuset(max_alloc_mem_g, uint64, 0);
	cuset(free_mem_g, uint64, 0);
	find_max_alloc_k<<<1, bs>>>();
	cucheck(cudaGetLastError());
	cucheck(cudaStreamSynchronize(0));
	find_free_mem_k<<<divup(MAX_NSBS, bs), bs>>>(ideal);
	cucheck(cudaGetLastError());
	cucheck(cudaStreamSynchronize(0));

	uint64 free_mem, max_alloc;
	//uint64 cuda_mem;
	cuget(&free_mem, free_mem_g);
	cuget(&max_alloc, max_alloc_mem_g);
	//cuget(&cuda_mem, cuda_mem_g);
	//	printf("free_mem = %lld, max_alloc = %lld, cuda_mem = %lld\n", 
	//			 free_mem, max_alloc, cuda_mem);
	return 1.0 - (double)max_alloc / free_mem;
}  // ha_extfrag
