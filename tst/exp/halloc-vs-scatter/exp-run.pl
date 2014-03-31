#! /usr/bin/env perl

# a script to perform experiment with settings
use POSIX;

#include common functions
do '../common.pl';

$memory = 512 * 1024 * 1024;
$group = 5;
$max_nthreads = 640 * 1024;
$mem_fraction = 0.4;
#$common = "-f$falloc -F$ffree -e$fexec -m$memory -g$group";

# private test
sub priv_test {
		$fexec = 0.75;
#		$ffree = 0.2;
		$ffree = 0.35;
		$total_niters = 128;
		$ocsv_name = "./exp-log-priv.csv";
		$common = "-e$fexec -m$memory -g$group";
		$OCSV = 100;
		open(OCSV, ">", $ocsv_name) 
				|| die "cannot open file $ocsv_name for writing";
		$oline = "allocator alloc_sz nallocs ntries thru\n";
		print $oline;
		print OCSV $oline;
		foreach $alloc_sz (16, 64) {
#		foreach $alloc_sz (16) {
#				for($nallocs = 1; $nallocs < 16; $nallocs++) {
#				for($ffree = 0.25; $ffree <= 0.35; $ffree += 0.01) {
				for($nthreads = 64 * 1024; $nthreads <= $max_nthreads; 
						$nthreads += 64 * 1024) {
#				for($ntries = 2; $ntries <= 32; $ntries += 2) {
#						$falloc = $ffree + $fexec - 0.01;
						$falloc = 0.9;
						$nallocs = $alloc_sz == 16 ? 4 : 1;
#						$nthreads = $max_nthreads;
#						foreach $allocator ("halloc", "scatter", "cuda") {
						foreach $allocator ("halloc", "scatter", "cuda") {
								$args = "-a$allocator -n$nthreads -l$nallocs -s$alloc_sz " . 
										"-f$falloc -F$ffree -e$fexec";
								# private speed
								$niters = 16;
								$ntries = $total_niters / $niters;
								if($allocator eq "cuda") {
										$ntries = 1;
								}
								runtest("throughput", $common, $args, "-i$niters -t$ntries");
								$oline = "$allocator $alloc_sz $nallocs $nthreads $thru_pair\n";
#								$oline = "$allocator $alloc_sz $nallocs $ffree $thru_pair\n";
								print OCSV $oline;
								print $oline;
						}
				}
		}  # foreach $alloc_sz		
		close OCSV;
} # sub priv_test

# spree test: fractions fixed, nthreads varies
sub spree_test {
		$ocsv_name = "./exp-log-spree.csv";
		$OCSV = 100;
		$falloc = 0.9; $ffree = 0.2; $fexec = 0.71;
		$total_niters = 16;
		$common = "-f$falloc -F$ffree -e$fexec -m$memory -g$group";
		open(OCSV, ">", $ocsv_name) 
				|| die "cannot open file $ocsv_name for writing";
		$oline = "allocator alloc_sz nallocs nthreads thru thru_malloc thru_free\n";
		print $oline;
		print OCSV $oline;
		foreach $alloc_sz (16, 64) {
				for($nthreads = 64 * 1024; $nthreads <= $max_nthreads;
						$nthreads += 64 *	1024) {
						$nallocs = $alloc_sz == 16 ? 4 : 1;
						foreach $allocator ("halloc", "scatter", "cuda") {
								$args = "-a$allocator -n$nthreads -l$nallocs -s$alloc_sz";
								# private speed
								$niters = 1;
								$ntries = $total_niters / $niters;
								runtest("throughput", $common, $args, "-i$niters -t$ntries");
								$oline = "$allocator $alloc_sz $nallocs $nthreads $thru_pair " 
										. "$thru_malloc $thru_free\n";
								print OCSV $oline;
								print $oline;
						}
				}
		}  # foreach $alloc_sz		
		close OCSV;
} # sub spree_test

# main
priv_test();
spree_test();
# run gnuplot
system('gnuplot', './exp-plot.gpl');
