#!/usr/bin/env bash

lscpu | grep ^Flags: | cut -f2 -d: | tr " " "\n" | grep ^avx

