#! /bin/bash

# usage: 
# ./run-test.sh <test-name> <test-args>

TEST_NAME=$1
shift 1
TEST_EXE=./bin/$TEST_NAME

# run the test
echo $TEST_EXE $@
$TEST_EXE $@
