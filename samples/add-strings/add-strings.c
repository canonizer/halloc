/** @file grid-points.cu a test where grid points are sorted into a grid */

#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

int divup(int a, int b) { return a / b + (a % b ? 1 : 0); }

typedef unsigned long long int uint64;

/** a random value in [a, b] range */
int random2(int a, int b) {
	//return a + random() % (b - a + 1);
	return a;
}

/** an array filled with random values in [a, b] range, with contiguous groups
		of p values starting at p being the same */
void random_array(int *arr, size_t n, int p, int a, int b) {
	int v = 0;
	for(size_t i = 0; i < n; i++) {
		if(i % p == 0)
			v = random2(a, b);
		arr[i] = v;
	}
}

void alloc_strs(char **strs, const int *lens, int n) {
#pragma omp parallel for
	for(int i = 0; i < n; i++) {
		int l = lens[i];
		char *str = (char *)malloc((l + 1) * sizeof(char));
		for(int j = 0; j < l; j++)
			str[j] = ' ';
		str[l] = 0;
		strs[i] = str;
	}
}  //alloc_strs

void free_strs(char ** strs, int n) {
	#pragma omp parallel for
	for(int i = 0; i < n; i++)
		free(strs[i]);
}  // free_strs

void add_strs(char ** __restrict__ c, char **a, char **b, int n) {
#pragma omp parallel for
	for(int i = 0; i < n; i++) {
		const char *sa = a[i], *sb = b[i];
		int la = strlen(sa), lb = strlen(sb), lc = la + lb;
		char *sc = (char *)malloc((lc + 1) * sizeof(char));
		strcpy(sc, sa);
		strcpy(sc + la, sb);
		c[i] = sc;
	}
}  // add_strs

#define MIN_LEN 31
#define MAX_LEN 31
#define PERIOD 32

/** a test for string addition on CPU */
void string_test_cpu(int n, int print) {
	int min_len = MIN_LEN;
	int max_len = MAX_LEN;
	int period = PERIOD;
	// string lengths on host and device
	int *h_la = 0, *h_lb = 0;
	size_t l_sz = n * sizeof(int), s_sz = n * sizeof(char *);
	h_la = (int *)malloc(l_sz);
	h_lb = (int *)malloc(l_sz);
	random_array(h_la, n, period, min_len, max_len);
	random_array(h_lb, n, period, min_len, max_len);

	// string arrays
	char **h_sa, **h_sb, **h_sc;
	h_sa = (char **)malloc(s_sz);
	h_sb = (char **)malloc(s_sz);
	h_sc = (char **)malloc(s_sz);

	// allocate strings
	double t1, t2;
	t1 = omp_get_wtime();
	alloc_strs(h_sa, h_la, n);
	alloc_strs(h_sb, h_lb, n);
	t2 = omp_get_wtime();
	//printf("t1 = %lf, t2 = %lf\n", t1, t2);
	if(print) {
		double t = (t2 - t1) / 2;
		printf("CPU allocation time: %4.2lf ms\n", t * 1e3);
		printf("CPU allocation performance: %4.2lf Mstrings/s\n", n / t * 1e-6);
	}

	//concatenate strings
	t1 = omp_get_wtime();
	add_strs(h_sc, h_sa, h_sb, n);
	t2 = omp_get_wtime();
	if(print) {
		double t = t2 - t1;
		printf("CPU concatenation time: %4.2lf ms\n", t * 1e3);
		printf("CPU concatenation performance: %4.2lf Mstrings/s\n", n / t * 1e-6);
	}

	// free strings
	t1 = omp_get_wtime();
	free_strs(h_sa, n);
	free_strs(h_sb, n);
	free_strs(h_sc, n);
	t2 = omp_get_wtime();
	if(print) {
		double t = (t2 - t1) / 3;
		//double t = (t2 - t1) / 2;
		printf("CPU freeing time: %4.2lf ms\n", t * 1e3);
		printf("CPU freeing performance: %4.2lf Mstrings/s\n", n / t * 1e-6);
	} 
	// free the rest
	free(h_sa);
	free(h_sb);
	free(h_sc);
	free(h_la);
	free(h_lb);
}  // string_test_cpu


int main(int argc, char **argv) {
	//srandom((int)time(0));
	size_t memory = 512 * 1024 * 1024;
	printf("==============================\n");
	// CPU test
	string_test_cpu(10000, 0);
	string_test_cpu(500000, 1);
	//ha_shutdown();
}  // main
