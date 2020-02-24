import cv2
import numpy as np
import os
import argparse
from os.path import isfile, join

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    #files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #for sorting the file names properly
    files = sorted(os.listdir(pathIn))
    for i in range(len(files)):
        if(files[i] == '.DS_Store'):
            continue
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--path_frames", help="path to the frames folder")
    ap.add_argument("-o", "--path_vid", help="path to the video output to save")

    args = vars(ap.parse_args())

    pathIn= args["path_frames"]
    pathOut = args["path_vid"]
    fps = 25.0
    convert_frames_to_video(pathIn, pathOut, fps)
