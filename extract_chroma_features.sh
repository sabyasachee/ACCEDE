#!/bin/bash
filename='../movie_audio_mono/'$1'.wav'
outfilename='../movie_audio_results_chroma/'$1'_features.txt'
../compar/./SMILExtract -C ../compar/config/chroma_fft.conf -I $filename -O $outfilename
