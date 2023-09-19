#!/bin/sh

for START in {0..50000..10000}
do
    STOP=$(($START + 10000))
    ./calc_observables $START $STOP
done

