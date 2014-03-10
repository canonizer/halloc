/** @file utils.cu utility function implementation */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint max_prime_below(uint n, uint nb) {
	for(uint p = n - 1; p >= 3; p--) {
		uint max_d = (uint)floor(sqrt(p));
		bool is_prime = true;
		for(uint d = 2; d <= max_d; d++)
			if(p % d == 0) {
				is_prime = false;
				break;
			}
		if(is_prime && n % p && nb % p)
			return p;
	}
	// if we are here, we can't find prime; exit with failure
	fprintf(stderr, "cannot find prime below %d not dividing %d\n", n, n);
	exit(-1);
	return ~0;
}  // max_prime_below
