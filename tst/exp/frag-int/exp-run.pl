#! /usr/bin/env perl

# data for internal fragmentation plot

# 2, 3
@alloc_szs = (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536,
							2048, 3072);
# 2, 3, 5
#@alloc_szs = (16, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256, 320, 384, 512,
#							640, 768, 1024, 1280, 1536,	2048, 2560, 3072);
# 2, 3, 5, 7
#@alloc_szs = (16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256,
#							320, 384, 448, 512,	640, 768, 896, 1024, 1280, 1536, 1792, 2048,
#							2560, 3072);
# 2, 3, 5, 7, 9
#@alloc_szs = (16, 24, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128, 144, 160, 192, 224, 256,
#							288, 320, 384, 448, 512, 576, 640, 768, 896, 1024, 1152, 1280,
#							1536, 1792, 2048, 2304,	2560, 3072);
#@alloc_szs = (16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224,
#							256, 320, 384, 448,	512, 640, 768, 896,
#							1024, 1280, 1536, 1792,	2048, 2560, 3072);
#@alloc_szs = (16, 24, 32, 40, 48, 56, 64, 80,	96, 112, 128, 144, 160, 176, 
#							192, 224, 256, 288, 320, 352, 384, 448, 512, 576, 640, 704,
#							768, 896, 1024, 1280, 1408, 1536, 1792, 2048, 2304, 2560, 3072);
#@alloc_szs = (16, 24, 32, 40, 48, 56, 64, 72, 80, 88,	96, 104, 112, 120, 
#							128, 144, 160, 176, 192, 208, 224, 240, 
#							256, 288, 320, 352, 384, 416, 448, 480, 
#							512, 576, 640, 704,	768, 832, 896, 960, 
#							1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920,
#							2048, 2304, 2560, 2816, 3072);
sub find_alloc_sz {
		my $sz = $_[0];
		foreach $alloc_sz (@alloc_szs) {
				if($alloc_sz >= $sz) {
						return $alloc_sz;
				}
		}
}  # find_alloc_sz

$min_sz = 16;
$max_sz = 3072;
$step_sz = 8;

$OCSV = 100;
$ofile = "./exp-log.csv";
open(OCSV, ">", $ofile) || die "cannot open $ofile for writing";
$oline = "nbytes block_frag cum_frag cum_frag2\n";
print OCSV $oline;
#print $oline;
print "$#alloc_szs sizes\n";

$sum_frag = 0;
$sum_alloc_sz = 0;
$sum_overhead = 0;
$n = 1;

for($sz = $min_sz; $sz <= $max_sz; $sz += $step_sz) {
		$alloc_sz = find_alloc_sz($sz);
		$overhead = $alloc_sz - $sz;
		$sum_overhead += $overhead;
		$sum_alloc_sz += $alloc_sz;
		$block_frag = $overhead / $alloc_sz;
		$sum_frag += $block_frag;
		$cum_frag = $sum_frag / $n;
		$cum_frag2 = $sum_overhead / $sum_alloc_sz;
		$n++;
		$oline = "$sz $block_frag $cum_frag $cum_frag2\n";
		print OCSV $oline;
#		print $oline;
}  # for($sz)

close OCSV;
system('gnuplot exp-plot.gpl');
