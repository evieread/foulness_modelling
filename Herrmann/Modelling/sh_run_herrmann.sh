#!/bin/bash


# Maximum frequency content of your synthetics is calculated using
# fmax = (number_samples / 2.0) / (number_samples * sampling_rate)
# number_samples - column 3 in distance file
# sampling_rate - column 2 in distance file

# Length of synthetics = number_samples * sampling_rate

# The greater the fmax and length of synthetics the longer it takes to calculate

# The purpose of this program is to generate a data file for use by the wavenumber integration programs.
# -M model
# -d station file
# -HS source depth in km
# -HR receiver depth in km
# -EQEX Compute earthquake/explosion Green’sfunction
hprep96 -M foulness_structure -d foulness_dist.station -HS 0.002 -HR 0.001 -EQEX

# This ctually creates the Greens functionss
hspec96

# This convolves the response with the source time function
# -i Dirac Delta function≠
# -D Output is ground displacement (cm)
# -V Output is ground velocity (cm/s)
# -A Output is ground acceleration
hpulse96 -p -l 1 -V > hpulse96.out

# This changes the trace data stream from Green’sfunctions to a three component time history.
# -M0 Moment Seismic moment in units of dyne-cm (1Nm = 1000 dyne-cm)
# -E Explosion 
# -A Az Source to Station Azimuth 
# -B Baz Station to Source azimuth
fmech96  -E -M0 1.0e13 -A 0.0 -B 180.0 < hpulse96.out > file96

# This program converts a file96 trace file to single trace SACfiles
f96tosac -B < file96


sactosac -m *.sac

