import cv2
import cv2.cv as cv
from load import movies

def save_fps():
	with open("../movie_results/fps.txt", "w") as fw:
		fw.write("movie\tfps\n")
		for movie in movies:
			filename = "../continuous-movies/" + movie + ".mp4"
			cam = cv2.VideoCapture(filename)
			fps = cam.get(cv.CV_CAP_PROP_FPS)
			fw.write("%s\t%f\n" % (movie, fps))
			cam.release()
			cv2.destroyAllWindows()

# save_fps()