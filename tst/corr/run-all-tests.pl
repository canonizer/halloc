#! /usr/bin/env perl

# a script to run all test for halloc

use POSIX;

$ntests = 0;
$nsuccesses = 0;

sub runtest {
		system("./run-test.sh", @_);
		if($? >> 8 == 0) {
				$nsuccesses++;
		}				
		$ntests++;
}  # runtest

# correctness memory allocation test; over all sizes, allocate/free 25% memory
# for each small size, and 12.5% memory for each large size
$memory = 512 * 1024 * 1024;
$step = 8;
for($alloc_sz = 16; $alloc_sz <= 512 * 1024; $alloc_sz += $step) {
		$fraction = $alloc_sz <= 2 * 1024 ? 0.25 : 0.125;
		$nthreads = floor($fraction * $memory / $alloc_sz);
		if($nthreads == 0) {
				next;
		}
		runtest("checkptr", "-l1", "-t4", "-m$memory", "-s$alloc_sz", 
						"-n$nthreads");
		# modify step
		if($alloc_sz >= 1024 * 1024) {
				$step = 1024 * 1024;
		} elsif($alloc_sz >= 128 * 1024) {
				$step = 128 * 1024;
		} elsif($alloc_sz >= 16 * 1024) {
				$step = 16 * 1024;
		} elsif($alloc_sz >= 2 * 1024) {
				$step = 2 * 1024;
		} elsif($alloc_sz >= 256) {
				$step = 256;
		} elsif($alloc_sz >= 64) {
				$step = 16;
		} else {
				$step = 8;
		}
}  # for($step)

# free slabs test - to ensure that slabs are freed correctly
runtest("freeslabs", "-m$memory");

# print the total count
$nfails = $ntests - $nsuccesses;
print "tests: $ntests TOTAL, $nsuccesses SUCCEEDED, $nfails FAILED\n";
