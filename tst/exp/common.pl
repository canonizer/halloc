#! /usr/bin/env perl


$device = 1;

# runs, sets exernal variables to data extracted from test run; 
# negative values mean that no data has been extracted
sub runtest {
		my $test = $_[0];
		shift @_;
		$test = "../../perf/bin/phase-$test";
		$args = join ' ', @_;
		#print $args;
		my @res = `$test -D$device $args`;
		shift @res;
		#print @res;
		# set standard variables to undefined
		#throughput
		$thru_malloc = -1;
		$thru_free = -1;
		$thru_pair = -1;
		#speed; note that there's no "free speed"
		$speed_malloc = -1;
		$speed_pair = -1;
		#latency: note that there's no pair latency
		$lat_malloc_min = -1;
		$lat_malloc_max = -1;
		$lat_malloc_avg = -1;
		$lat_free_min = -1;
		$lat_free_max = -1;
		$lat_free_avg = -1;
		# analyze result lines
		
		foreach $line (@res) {
				my @fields = split ' ', $line;
				#print (join ',', @fields);
				my $is_malloc = grep /malloc/, @fields;
				my $is_free = grep /free/, @fields;
				my $is_pair = grep /pair/, @fields;
				my $is_thru = grep /throughput/, @fields;
				my $is_speed = grep /speed/, @fields;
				my $is_lat = grep /latency/, @fields;
				my $is_avg = grep /avg/, @fields;
				my $is_min = grep /min/, @fields;
				my $is_max = grep /max/, @fields;
				#print $is_pair, $is_thru, $is_malloc, "\n";
				if($is_thru) {
						if($is_malloc) {
								$thru_malloc = $fields[2];
						} elsif($is_free) {
								$thru_free = $fields[2];
						} elsif($is_pair) {
								$thru_pair = $fields[2];
						}
				} elsif($is_speed) {
						if($is_malloc) {
								$speed_malloc = $fields[2];
						} elsif($is_pair) {
								$speed_pair = $fields[2];
						}
				} elsif($is_lat) {
						if($is_malloc) {
								if($is_min) {
										$lat_malloc_min = $fields[3];
								} elsif($is_max) {
										$lat_malloc_max = $fields[3];
								} elsif($is_avg) {
										$lat_malloc_avg = $fields[3];
								}
						} elsif($is_free) {
								if($is_min) {
										$lat_free_min = $fields[3];
								} elsif($is_max) {
										$lat_free_max = $fields[3];
								} elsif($is_avg) {
										$lat_free_avg = $fields[3];
								}
						}
				}
		}  # foreach $line
}  # sub runtest
