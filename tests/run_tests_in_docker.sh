#!/bin/bash

# It may be useful to run the tests in docker to track down issues using a standardised environment
# This script will build the docker image, run the tests and copy the output to the local machine

rm -rf docker_test_output
docker build .. -t topsy
docker run --name running-tests topsy -c 'pytest'
docker cp running-tests:/app/tests/output ./docker_test_output
docker rm running-tests

