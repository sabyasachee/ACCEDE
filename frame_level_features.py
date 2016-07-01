#!/usr/bin/env python
# Krishna Somandepalli - 12/28/2015
import numpy as np
import cv2
import cv2.cv as cv
import sys, getopt, os
from os import getcwd, listdir
from os.path import join, isfile, expanduser
import sys

#####################   		Main program starts here

def get_flow_img(hsv, frame1_gs, gs_img):
	
	#hsv = np.zeros_like(frame1_gs)
	hsv[...,1] = 255

	flow = cv2.calcOpticalFlowFarneback(frame1_gs, gs_img, 0.5, 3, 9, 3, 5, 1.2, 0)
	#     frame1_gs = gs_img

	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
	hsv[...,0] = ang*180/np.pi/2
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
	flow_rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
	flow_gs = cv2.cvtColor(flow_rgb, cv2.COLOR_BGR2GRAY)
					        
	return flow_gs

def get_img_luma(img):
	frame_lum = 0.27*img[:,:,-1] + 0.67*img[:,:,1] + 0.06*img[:,:,0]
	lum_mean = np.mean(frame_lum.flatten())
	return lum_mean

def frame_level_features(fileID):

    #Load the scenes file

    
    #read videos
	filename = "../video/ACCEDE" + str(fileID).zfill(5) + ".mp4"
	outname = "ACCEDE" + str(fileID).zfill(5)
	DOWNSAMPLE=5
    #Read the video file
	cam  = cv2.VideoCapture(filename)
	fps=cam.get(cv.CV_CAP_PROP_FPS)	#Read the file frame rate
	print fps

	frame_number=0
	current_scene_number=0
	frames_this_shot=0

    #color = np.random.randint(0,255,(1000,3))
    #feature_params = dict( maxCorners = 300, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
    #lk_params = dict(winSize = old_gray.shape, maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	luma=[]
	intensity_avg=[]
	flow_intensity_avg=[0]

    #frame_downsample_ref = 5
	while(cam.isOpened()):
		ret, img = cam.read()
		if img is not None:	
			if not(ret):
				print "Skipping frame.."
				continue
			
			## Accumulate stuff for this shot
			#frames_this_shot+=1
			img_height, img_width, ch = img.shape
			hsv_flow = np.zeros_like(img)
			if not frame_number%100 : 
				print frame_number
			if frame_number == 0:
				#ggb2gs
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				gray = cv2.equalizeHist(gray)
				flow_ref_img = gray.copy()

				lum_mean = get_img_luma(img)
				## Compute per frame luminance
				luma.append(lum_mean)

				## Compute intensity average
				img_mean = cv2.mean(gray)
				intensity_avg.append(img_mean[0])
			
			## Compute dense optical flow
			elif frame_number >= 1:
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				gray = cv2.equalizeHist(gray)
				flow_input_img = gray.copy()
				flow_gs = get_flow_img(hsv_flow, flow_ref_img, flow_input_img)
				#flow_ref_img = flow_input_img
				flow_intensity_avg.append(cv2.mean(flow_gs)[0])

				lum_mean = get_img_luma(img)
				## Compute per frame luminance
				luma.append(lum_mean)

				## Compute RGB intensity average
				img_mean = cv2.mean(gray)
				intensity_avg.append(img_mean[0])
				flow_ref_img = gray
				
			frame_number+=1
		else: 
			cam.release()
			cv2.destroyAllWindows()
#save all low level features
	np.save("../results/luminance/"+outname+'_luma.npy',luma)
	np.save("../results/intensity/"+outname+'_intensity.npy', intensity_avg)
	np.save("../results/optical_flow/"+outname+'_flow.npy', flow_intensity_avg)

def frame_level_features_movies(filename):

    #Load the scenes file
    #read videos
	# filename = "../video/ACCEDE" + str(fileID).zfill(5) + ".mp4"
	print filename
	outname = filename

	DOWNSAMPLE=5
    #Read the video file
	cam  = cv2.VideoCapture("../continuous-movies/" + filename + ".mp4")
	fps=cam.get(cv.CV_CAP_PROP_FPS)	#Read the file frame rate
	print fps

	frame_number=0
	current_scene_number=0
	frames_this_shot=0

    #color = np.random.randint(0,255,(1000,3))
    #feature_params = dict( maxCorners = 300, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
    #lk_params = dict(winSize = old_gray.shape, maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	luma=[]
	intensity_avg=[]
	flow_intensity_avg=[0]

    #frame_downsample_ref = 5
	while(cam.isOpened()):
		ret, img = cam.read()
		if img is not None:	
			if not(ret):
				print "Skipping frame.."
				continue
			
			## Accumulate stuff for this shot
			#frames_this_shot+=1
			img_height, img_width, ch = img.shape
			hsv_flow = np.zeros_like(img)
			if not frame_number%100 : 
				print frame_number
			if frame_number == 0:
				#ggb2gs
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				gray = cv2.equalizeHist(gray)
				flow_ref_img = gray.copy()

				lum_mean = get_img_luma(img)
				## Compute per frame luminance
				luma.append(lum_mean)

				## Compute intensity average
				img_mean = cv2.mean(gray)
				intensity_avg.append(img_mean[0])
			
			## Compute dense optical flow
			elif frame_number >= 1:
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				gray = cv2.equalizeHist(gray)
				flow_input_img = gray.copy()
				flow_gs = get_flow_img(hsv_flow, flow_ref_img, flow_input_img)
				#flow_ref_img = flow_input_img
				flow_intensity_avg.append(cv2.mean(flow_gs)[0])

				lum_mean = get_img_luma(img)
				## Compute per frame luminance
				luma.append(lum_mean)

				## Compute RGB intensity average
				img_mean = cv2.mean(gray)
				intensity_avg.append(img_mean[0])
				flow_ref_img = gray
				
			frame_number+=1
		else: 
			cam.release()
			cv2.destroyAllWindows()
#save all low level features
	np.save("../movie_results/luminance/"+outname+'_luma.npy',luma)
	np.save("../movie_results/intensity/"+outname+'_intensity.npy', intensity_avg)
	np.save("../movie_results/optical_flow/"+outname+'_flow.npy', flow_intensity_avg)

def frame_level_features_test(filename):

	outname = filename.strip().split('/')[-1].split('.')[0]
	print outname
	DOWNSAMPLE=5
	cam  = cv2.VideoCapture(filename)
	fps=cam.get(cv.CV_CAP_PROP_FPS)	#Read the file frame rate
	print fps

	frame_number=0
	current_scene_number=0
	frames_this_shot=0

    #color = np.random.randint(0,255,(1000,3))
    #feature_params = dict( maxCorners = 300, qualityLevel = 0.3, minDistance = 7, blockSize = 7 )
    #lk_params = dict(winSize = old_gray.shape, maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	luma=[]
	intensity_avg=[]
	flow_intensity_avg=[0]

    #frame_downsample_ref = 5
	while(cam.isOpened()):
		ret, img = cam.read()
		if img is not None:	
			if not(ret):
				print "Skipping frame.."
				continue
			
			## Accumulate stuff for this shot
			#frames_this_shot+=1
			img_height, img_width, ch = img.shape
			hsv_flow = np.zeros_like(img)
			if not frame_number%100 : 
				print frame_number
			if frame_number == 0:
				#ggb2gs
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				gray = cv2.equalizeHist(gray)
				flow_ref_img = gray.copy()

				lum_mean = get_img_luma(img)
				## Compute per frame luminance
				luma.append(lum_mean)

				## Compute intensity average
				img_mean = cv2.mean(gray)
				intensity_avg.append(img_mean[0])
			
			## Compute dense optical flow
			elif frame_number >= 1:
				gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				gray = cv2.equalizeHist(gray)
				flow_input_img = gray.copy()
				flow_gs = get_flow_img(hsv_flow, flow_ref_img, flow_input_img)
				#flow_ref_img = flow_input_img
				flow_intensity_avg.append(cv2.mean(flow_gs)[0])

				lum_mean = get_img_luma(img)
				## Compute per frame luminance
				luma.append(lum_mean)

				## Compute RGB intensity average
				img_mean = cv2.mean(gray)
				intensity_avg.append(img_mean[0])
				flow_ref_img = gray
				
			frame_number+=1
		else: 
			cam.release()
			cv2.destroyAllWindows()
#save all low level features
	np.save("../test_results/luminance/"+outname+'_luma.npy',luma)
	np.save("../test_results/intensity/"+outname+'_intensity.npy', intensity_avg)
	np.save("../test_results/optical_flow/"+outname+'_flow.npy', flow_intensity_avg)

if __name__ == '__main__':
	filename = sys.argv[1]
	frame_level_features_test(filename)