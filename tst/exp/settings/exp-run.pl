#! /usr/bin/env perl

# a script to perform experiment with settings
use POSIX;

#include common functions
do '../common.pl';

$falloc = 0.95;
$ffree = 0.05;
$fexec = 0.91;
$nthreads = 512 * 1024;
$memory = 512 * 1024 * 1024;
$group = 5;
$niters = 31;
$ntries = 16;

$common = "-f$falloc -F$ffree -e$fexec -n$nthreads -m$memory " .
	"-g$group -i$niters -t$niters";
$ocsv_name = "./exp-log.csv";
$OCSV = 100;
open(OCSV, ">", $ocsv_name) 
		|| die "cannot open file $ocsv_name for writing";
$oline = "slab_size busy nallocs alloc_sz throughput speed\n";
print OCSV $oline;
print $oline;
foreach $slab_size (20, 21, 22, 23) {
#foreach $slab_size (22) {
#		foreach $busy (0.75, 0.835, 0.9, 0.95) {
#		foreach $busy (0.835) {
		for($busy = 0.745; $busy <= 0.955; $busy += 0.01) {
				foreach $alloc_sz (16, 256) {
#				foreach $alloc_sz (16) {
						my $nallocs = $alloc_sz == 16 ? 4 : 1;
						runtest("throughput", $common, "-b$slab_size", "-B$busy",
										"-s$alloc_sz", "-l$nallocs");
						my $slab_sz = 2 ** ($slab_size - 20);
						$oline = 
								"$slab_sz $busy $nallocs $alloc_sz $thru_pair $speed_pair\n";
						print OCSV $oline;
						print $oline;
				}  # foreach alloc_sz
		}  # foreach busy
}  # foreach slab_size

close OCSV;
# run gnuplot
system('gnuplot', './exp-plot.gpl');
