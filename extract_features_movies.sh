#!/bin/bash
filename='../continuous-movies/'$1'.mp4'
outfilename='../movie_audio_mono/'$1'.wav'
audiooutfilename='../movie_audio_results/'$1'_features.txt'
echo $filename
cp -f $filename ~/Documents/temp$1.mp4
avconv -i ~/Documents/temp$1.mp4 -vn -f wav ~/Documents/temp_audio$1.wav -y
sox ~/Documents/temp_audio$1.wav -c 1 $outfilename
python frame_level_features.py $1
../compar/./SMILExtract -C ../compar/config/IS13_ComParE_Voc.conf -I $outfilename -O $audiooutfilename
rm ~/Documents/temp$1.mp4
rm ~/Documents/temp_audio$1.wav