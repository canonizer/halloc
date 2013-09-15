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
$total_niters = 512;
$common = "-f$falloc -F$ffree -e$fexec -m$memory -g$group";

# running a speed test
sub run_speedtest {
		# spree speed
		$ntries = $total_niters;
		runtest("throughput", $common, $_[0], "-i1 -t$ntries");
		$spree_speed = $speed_pair;
		$spree_speed_malloc = $speed_malloc;
		$spree_thru = $thru_pair;
		$spree_thru_malloc = $thru_malloc;
		# private speed
		$niters = 32;
		$ntries = $total_niters / $niters;
		runtest("throughput", $common, $_[0], "-i$niters -t$ntries");
		$priv_speed = $speed_pair;
		$priv_thru = $thru_pair;
} # run_speedtest

# single-size test
sub single_size {
		@alloc_szs = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536,
									2048, 3072);
		$ocsv_name = "./exp-log-single.csv";
		$OCSV = 100;
		open(OCSV, ">", $ocsv_name) 
				|| die "cannot open file $ocsv_name for writing";
		$oline = "block priv_speed priv_thru spree_speed spree_speed_malloc " . 
				"speed_thru speed_thru_malloc\n";
		print $oline;
		print OCSV $oline;
		foreach $alloc_sz (@alloc_szs) {
				$nallocs = 1;
				if($alloc_sz <= 64) {
						$nallocs = 4;
				} elsif($alloc_sz <= 128) {
						$nallocs = 2;
				}
				$nthreads = floor($mem_fraction * $memory / ($alloc_sz * $nallocs));
				$nthreads = $nthreads > $max_nthreads ? $max_nthreads : $nthreads;
				run_speedtest("-n$nthreads -l$nallocs -s$alloc_sz");
				$oline = "$alloc_sz $priv_speed $priv_thru " . 
						"$spree_speed $spree_speed_malloc $spree_thru $spree_thru_malloc\n";
				print OCSV $oline;
				print $oline;		
		}  # foreach $alloc_sz
		
		close OCSV;
} # sub single_size

# combined-size tests
sub combi_size {
		@min_alloc_szs = (8, 8, 8, 8);
		@max_alloc_szs = (32, 64, 256, 3072);
		$ocsv_name = "./exp-log-combi.csv";
		$OCSV = 100;
		open(OCSV, ">", $ocsv_name) 
				|| die "cannot open file $ocsv_name for writing";
		$oline = "block priv_speed priv_thru spree_speed spree_speed_malloc " . 
				"speed_thru speed_thru_malloc\n";
		print $oline;
		print OCSV $oline;
		for($isz = 0; $isz < @min_alloc_szs; $isz++) {
				$min_sz = $min_alloc_szs[$isz];
				$max_sz = $max_alloc_szs[$isz];
				$nallocs = 1;
				if($max_sz <= 64) {
						$nallocs = 4;
				}
				$avg_sz = ($min_sz + $max_sz) / 2;
				$nthreads = floor($mem_fraction * $memory / ($avg_sz * $nallocs));
				$nthreads = $nthreads > $max_nthreads ? $max_nthreads : $nthreads;
				run_speedtest("-n$nthreads -l$nallocs -s$min_sz -S$max_sz");
				$oline = "$min_sz..$max_sz $priv_speed $priv_thru " . 
						"$spree_speed $spree_speed_malloc $spree_thru $spree_thru_malloc\n";
				print OCSV $oline;
				print $oline;		
		}  # foreach $alloc_sz

		close OCSV;
} # sub combi_size


# main
single_size();
combi_size();
# run gnuplot
system('gnuplot', './exp-plot.gpl');
