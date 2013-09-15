#! /bin/bash

cd scaling
./exp-run.pl
cd ..

cd settings
./exp-run.pl
cd ..

cd speed
./exp-run.pl
cd ..

cd halloc-vs-scatter
./exp-run.pl
cd ..
