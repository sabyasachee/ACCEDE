function saliencyMeasures()
parfor i=0:9799
    filename = strcat('Documents/video/ACCEDE', sprintf('%05d.mp4', i));
    outfilename = strcat('Documents/results/saliency/ACCEDE', sprintf('%05d_saliency.txt', i));
    fileID = fopen(outfilename, 'w');
    fprintf(fileID, 'frame_number\t0.2\t0.5\t0.7\n');
    disp(filename);
    video = VideoReader(filename);
    nframes = video.NumberOfFrames;
    width = video.Width;
    heigth = video.Height;
    area = width * heigth;
    for j=1:nframes
        frame = read(video, j);
        [L, a, b] = RGB2Lab(frame);
        salMat = saliencyMeasure({L, a, b});
        sum_2 = sum(salMat(:)>0.2);
        sum_5 = sum(salMat(:)>0.5);
        sum_7 = sum(salMat(:)>0.7);
        ratio_2 = sum_2/area;
        ratio_5 = sum_5/area;
        ratio_7 = sum_7/area;
        fprintf(fileID, '%d\t%f\t%f\t%f\n', j-1, ratio_2, ratio_5, ratio_7);
        fprintf('Frame_number %d done\n', j);
    end
    fclose(fileID);
end
end