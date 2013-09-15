#! /usr/bin/env perl

# a script to perform experiment with settings
use POSIX;

#include common functions
do '../common.pl';

$falloc = 0.95;
$ffree = 0.05;
$fexec = 0.91;
$memory = 512 * 1024 * 1024;
$group = 5;
$max_nthreads = 1024 * 1024;
$mem_fraction = 0.4;
$common = "-f$falloc -F$ffree -e$fexec -m$memory -g$group";

# throughput test
sub thru_test {
		$total_niters = 512;
		$ocsv_name = "./exp-log-thru.csv";
		$OCSV = 100;
		open(OCSV, ">", $ocsv_name) 
				|| die "cannot open file $ocsv_name for writing";
		$oline = "alloc_sz nallocs nthreads priv_pair spree_pair spree_malloc spree_free\n";
		print $oline;
		print OCSV $oline;
		for($nthreads = 32; $nthreads <= 1024 * 1024; $nthreads *= 2) {
				foreach $alloc_sz (16, 256) {
						foreach $nallocs (1, 4) {
								if($nallocs == 4 && $alloc_sz > 64) {
										next;
								}
								# spree test
								$ntries = $total_niters;
								$args = "-n$nthreads -l$nallocs -s$alloc_sz";
								runtest("throughput", $common, $args, "-i1 -t$ntries");
								$spree_pair = $thru_pair;
								$spree_malloc = $thru_malloc;
								$spree_free = $thru_free;
								# private test
								$niters = 32;
								$ntries = $total_niters / $niters;
								runtest("throughput", $common, $args, "-i$niters -t$ntries");
								$priv_pair = $thru_pair;
								$oline = "$alloc_sz $nallocs $nthreads $priv_pair " . 
										"$spree_pair $spree_malloc $spree_free\n";
								print OCSV $oline;
								print $oline;
						}  # foreach $nallocs
				}  # foreach $alloc_sz
		}  # for($nthreads)
		close OCSV;
} # sub thru_test

# latency test
sub lat_test {
		$total_niters = 128;
		$ocsv_name = "./exp-log-lat.csv";
		$OCSV = 100;
		open(OCSV, ">", $ocsv_name) 
				|| die "cannot open file $ocsv_name for writing";
		$oline = "alloc_sz nallocs nthreads malloc_min malloc_avg malloc_max " . 
				"free_min free_avg free_max\n";
		print $oline;
		print OCSV $oline;
		for($nthreads = 1; $nthreads <= 1024 * 1024; $nthreads *= 2) {
				foreach $alloc_sz (16, 256) {
						$nallocs = 1;
						#foreach $nallocs (1, 4) {
						#		if($nallocs == 4 && $alloc_sz > 64) {
						#				next;
						#		}
								# private test
								$niters = 16;
								$ntries = $total_niters / $niters;
								$args = "-n$nthreads -l$nallocs -s$alloc_sz";
								runtest("latency", $common, $args, "-i$niters -t$ntries");
								$priv_pair = $thru_pair;
								$oline = "$alloc_sz $nallocs $nthreads " . 
										"$lat_malloc_min $lat_malloc_avg $lat_malloc_max " . 
										"$lat_free_min $lat_free_avg $lat_free_max\n";
								print OCSV $oline;
								print $oline;
						#}  # foreach $nallocs
				}  # foreach $alloc_sz
		}  # for($nthreads)
		close OCSV;
} # sub lat_test

# main 
thru_test();
lat_test();
# run gnuplot
system('gnuplot', './exp-plot.gpl');
