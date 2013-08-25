#! /bin/bash

# runs throughput test for all sizes
./run-test.sh throughput -s16 -l4
./run-test.sh throughput -s24 -l4
./run-test.sh throughput -s32 -l4
./run-test.sh throughput -s48 -l4
./run-test.sh throughput -s64 -l4
./run-test.sh throughput -s96 -l2
./run-test.sh throughput -s128 -l2
./run-test.sh throughput -s192 -l1
./run-test.sh throughput -s256 -l1
./run-test.sh throughput -s384 -l1 -n$((512*1024))
./run-test.sh throughput -s512 -l1 -n$((512*1024))
./run-test.sh throughput -s768 -l1 -n$((256*1024))
./run-test.sh throughput -s1024 -l1 -n$((256*1024))
./run-test.sh throughput -s1536 -l1 -n$((128*1024))
./run-test.sh throughput -s2048 -l1 -n$((128*1024))
./run-test.sh throughput -s3072 -l1 -n$((64*1024))

