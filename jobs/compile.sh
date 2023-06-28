#!/bin/sh

ml purge
ml CMake/3.23.1-GCCcore-11.3.0
ml CUDA/12.0.0

SRC_DIR=../src
BUILD_DIR=build
echo $SRC_DIR
cmake -S $SRC_DIR -B $BUILD_DIR
cmake --build $BUILD_DIR
