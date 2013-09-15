#! /usr/bin/env perl

# a script to perform experiment with settings
use POSIX;

#include common functions
do '../common.pl';

$memory = 512 * 1024 * 1024;
$group = 5;
$max_nthreads = 512 * 1024;
$mem_fraction = 0.4;
#$common = "-f$falloc -F$ffree -e$fexec -m$memory -g$group";

# private test
sub priv_test {
		$fexec = 0.61;
		$total_niters = 256;
		$ocsv_name = "./exp-log-priv.csv";
		$common = "-e$fexec -m$memory -g$group";
		$OCSV = 100;
		open(OCSV, ">", $ocsv_name) 
				|| die "cannot open file $ocsv_name for writing";
		$oline = "allocator alloc_sz nallocs ffree thru\n";
		print $oline;
		print OCSV $oline;
		foreach $alloc_sz (16, 64) {
				for($ffree = 0.20; $ffree <= 0.35; $ffree += 0.01) {
						$falloc = $ffree + $fexec - 0.01;
						$nallocs = $alloc_sz == 16 ? 4 : 1;
						$nthreads = $max_nthreads * $nallocs * $alloc_sz / (16 * 4);
						foreach $allocator ("halloc", "scatter") {
								$args = "-a$allocator -n$nthreads -l$nallocs -s$alloc_sz " . 
										"-f$falloc -F$ffree";
								# private speed
								$niters = 32;
								$ntries = $total_niters / $niters;
								runtest("throughput", $common, $args, "-i$niters -t$ntries");
								$oline = "$allocator $alloc_sz $nallocs $ffree $thru_pair\n";
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
		$falloc = 0.95; $ffree = 0.05; $fexec = 0.91;
		$total_niters = 16;
		$common = "-f$falloc -F$ffree -e$fexec -m$memory -g$group";
		open(OCSV, ">", $ocsv_name) 
				|| die "cannot open file $ocsv_name for writing";
		$oline = "allocator alloc_sz nallocs nthreads thru thru_malloc thru_free\n";
		print $oline;
		print OCSV $oline;
		foreach $alloc_sz (16, 64) {
				for($nthreads = 32 * 1024; $nthreads <= $max_nthreads;
						$nthreads += 32 *	1024) {
						$nallocs = $alloc_sz == 16 ? 4 : 1;
						foreach $allocator ("halloc", "scatter") {
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
