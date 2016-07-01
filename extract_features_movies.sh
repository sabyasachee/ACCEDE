#!/bin/bash
for i in `seq 0 9`;do
	echo $i
	name='MEDIAEVAL16_FM_0'$i
	filename='../MEDIAEVAL16-ContinuousPrediction-data/data/'$name'.mp4'
	outfilename='../test_audio_mono/'$name'.wav'
	audiooutfilename='../test_audio_results/'$name'_features.txt'
	chromaoutfilename='../test_chroma_results/'$name'_features.txt'
	cp -f $filename ~/Documents/temp_$name.mp4
	avconv -i ~/Documents/temp_$name.mp4 -vn -f wav ~/Documents/temp_audio_$name.wav -y
	sox ~/Documents/temp_audio_$name.wav -c 1 $outfilename
	python frame_level_features.py $filename
	../compar/./SMILExtract -C ../compar/config/IS13_ComParE_Voc.conf -I $outfilename -O $audiooutfilename
	../compar/./SMILExtract -C ../compar/config/chroma_fft.conf -I $outfilename -O $chromaoutfilename
	rm ~/Documents/temp_$name.mp4
	rm ~/Documents/temp_audio_$name.wav
done