#!/bin/sh

cmake -S /home/stan/ANN -B /home/stan/ANN/out/build
cd /home/stan/ANN/out/build/
make
cd /home/stan/ANN