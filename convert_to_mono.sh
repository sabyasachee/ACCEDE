#!/bin/bash
for i in `seq 0 9799`; do
	filename="../audio/ACCEDE"`printf %05d $i`".wav"
	outfilename="../audio_mono/ACCEDE"`printf %05d $i`".wav"
	printf "Working on $filename\n"
	sox $filename -c 1 $outfilename
done