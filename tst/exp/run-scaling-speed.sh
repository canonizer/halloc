#! /bin/bash


cd scaling
./exp-run.pl
#gnuplot ./exp-plot.gpl
cd ..

cd settings
./exp-run.pl
#gnuplot ./exp-plot.gpl
cd ..

cd speed
./exp-run.pl
#gnuplot ./exp-plot.gpl
cd ..
