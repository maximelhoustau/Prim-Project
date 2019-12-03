import cv2
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help = "path to the video file")
ap.add_argument("-i", "--image", help = "path to the image file saved")
ap.add_argument("-t", "--time", type=int, help = "time corresponding to the extracted frame")

args = vars(ap.parse_args())

vidcap = cv2.VideoCapture(args["video"])
time = args["time"]*1000
vidcap.set(cv2.CAP_PROP_POS_MSEC,time)      # just cue to 20 sec. position
success,image = vidcap.read()
if success:
    cv2.imwrite(args["image"], image)     # save frame as JPEG file
    cv2.imshow(str(time)+"sec",image)
    cv2.waitKey()
