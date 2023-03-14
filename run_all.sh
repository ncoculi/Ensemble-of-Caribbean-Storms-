#!/bin/bash
for storm in Allen Gilbert Iris Ivan Matthew Tomas David Gustav Irma Maria; do
    cd $storm;
    make clean;
    make all;
    python fgmax-to-ascii.py;
    cd ..;
done

