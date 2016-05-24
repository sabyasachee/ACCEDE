import dlib
import cv2
import cv2.cv as cv
from thread_module import process_multithread

detector = dlib.get_frontal_face_detector()

def face_detection(file_id):
    filename = "../video/ACCEDE" + str(file_id).zfill(5) + ".mp4"
    print filename
    cam = cv2.VideoCapture(filename)
    fps = cam.get(cv.CV_CAP_PROP_FPS)
    width = cam.get(cv.CV_CAP_PROP_FRAME_WIDTH)
    height = cam.get(cv.CV_CAP_PROP_FRAME_HEIGHT)
    print fps, width, height
    area = width*height
    frame_number = 0
    with open("../results/faces/ACCEDE" + str(file_id).zfill(5)+"_faces.txt", "w") as fw:
        fw.write("%d %d\n" % (width, height))
        fw.write("%f\n" % fps)
        fw.write("frame_number\tnumber_of_faces\t[left\ttop\tright\tbottom\tarea_percentage\tscore\ttype]\n")
        while cam.isOpened():
            ret, img = cam.read()
            if img is not None:
                if ret is None:
                    print "\tSkipping Frame"
                    continue
                if not frame_number % 100:
                    print "\tFrame Number %d" % frame_number
                dets, scores, idx = detector.run(img, 1)
                line = "%d\t%d\t" % (frame_number, len(dets))
                for i, d in enumerate(dets):
                    face_area = (d.right() - d.left())*(d.bottom() - d.top())
                    line += "%d\t%d\t%d\t%d\t%f\t%f\t%d\t" % (d.left(), d.top(), d.right(), d.bottom(), float(face_area * 100)/area, scores[i], idx[i])
                line += "\n"
                fw.write(line)
                frame_number += 1
            else:
                cam.release()
                cv2.destroyAllWindows()

process_multithread(face_detection)