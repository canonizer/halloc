#! /usr/bin/env perl

# a script to run all test for halloc

use POSIX;

$ntests = 0;
$nsuccesses = 0;
$device = 0;
$memory = 512 * 1024 * 1024;

sub runtest {
		system("./run-test.sh", @_, "-D$device", "-m$memory");
		if($? >> 8 == 0) {
				$nsuccesses++;
		}				
		$ntests++;
}  # runtest

# correctness memory allocation test; over all sizes, allocate/free 25% memory
# for each small size, and 12.5% memory for each large size
$step = 8;
for($alloc_sz = 16; $alloc_sz <= 32 * 1024; $alloc_sz += $step) {
		$fraction = $alloc_sz <= 2 * 1024 ? 0.25 : 0.125;
		$nthreads = floor($fraction * $memory / $alloc_sz);
		if($nthreads == 0) {
				next;
		}
		runtest("checkptr", "-l1", "-t4", "-s$alloc_sz", "-n$nthreads");
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

# probabilitized tests
$falloc = 0.5;
$ffree = 0.5;
$fexec = 0.75;
#foreach $group (10) {
foreach $group (0, 5, 10) {
		foreach $niters (1, 5) {
#		foreach $niters (1) {
				$ntries = $group == 1 ? 1024 : 16384;
				$ntries = ceil($ntries / $niters);
				@fixed_args = ("prob-checkptr", "-i$niters", "-t$ntries", "-f$falloc",
											 "-F$ffree", "-e$fexec", "-g$group");
				# small sizes (<= 64 bytes)
				$nthreads = 1024 * 1024;
				runtest(@fixed_args, "-l4", "-n$nthreads", "-s8", "-S64", "-duniform");
				# medium sizes (<= 256 bytes)
				runtest(@fixed_args, "-l1", "-n$nthreads", "-s8", "-S256", "-duniform");
				runtest(@fixed_args, "-l4", "-n$nthreads", "-s8", "-S256",	"-dexpequal");
				# large-size test (<= 3072 bytes)				
				$nthreads = 64 * 1024;
				runtest(@fixed_args, "-l1", "-n$nthreads", "-s8", "-S3072", "-duniform");
				$nthreads = 128 * 1024;
				runtest(@fixed_args, "-l4", "-n$nthreads", "-s8", "-S3072",	"-dexpequal");
		}
}

# print the total count
$nfails = $ntests - $nsuccesses;
print "tests: $ntests TOTAL, $nsuccesses SUCCEEDED, $nfails FAILED\n";
