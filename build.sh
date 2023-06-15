#!/usr/bin/env bash

BUILDDIR=build

conan install . --output-folder=$BUILDDIR --build=missing
pushd $BUILDDIR
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build .
popd
