#! /bin/bash

# usage: 
# ./run-test.sh <test-name> <test-args>

TEST_NAME=$1
shift 1
TEST_EXE=./bin/$TEST_NAME

# run the test
echo $TEST_EXE $@
$TEST_EXE $@

# analyze exit code
# TODO: add output coloring
TEST_EXIT=$?
if [ $TEST_EXIT == 0 ]; then
		echo "$TEST_NAME test PASSED"
		exit 0
else
		echo "$TEST_NAME test FAILED with exit code $TEST_EXIT"
		exit -1
fi
