from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.Apply_mask import apply_mask


start_time = T.time()
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
ap.add_argument("-d", "--display", type=int, default=1, help="minimum area size")

args = vars(ap.parse_args())

image_folder = "./images/"
video_folder = "./videos/"

timeline = []
motion = []
event = False
time = 0

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

# otherwise, we are reading from a video file
else:
        vs = cv2.VideoCapture(video_folder + args["video"])

# Get the fps and number of frame to get the time of the video
fps = vs.get(cv2.CAP_PROP_FPS)
count = 0

# loop over the frames of the video
frame = vs.read()[1]
while True:
        # grab the current frame 
        frame_t = vs.read()[1] 

        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame_t is None:
                break

        # resize the frame, convert it to grayscale, and blur it
        #frame = imutils.resize(frame, width=800)
        #frame_t = imutils.resize(frame_t, width=800)
        frame = apply_mask(frame) 
        frame_t = apply_mask(frame_t) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        gray_t = cv2.GaussianBlur(gray_t, (21, 21), 0)
        
        # compute motion frames
        frameDelta = cv2.absdiff(gray, gray_t)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=4)
    
        #Show the real time videos if display is not set at 0
        if (args["display"]):

            # show the frame and record if the user presses a key
            cv2.imshow("Frame", frame)
            cv2.imshow("Thresh", thresh)
            cv2.imshow("Frame Delta", frameDelta)

        # if the `q` key is pressed, break from the lop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break
            
        if key == ord("w"):
            cv2.imwrite(image_folder+"MD_frame.jpg", frame)
            cv2.imwrite(image_folder+"MD_thresh.jpg", thresh)
            cv2.imwrite(image_folder+"MD_delta.jpg", frameDelta)
        
        #Count for next frame
        count += 1

        #Add time and motion counter of this frame to their vector in order to plot
        timeline.append(count/fps)  
        motion.append(np.count_nonzero(thresh))
        #update frame
        frame = frame_t

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()

print("Execution time : "+str(T.time() - start_time))

#Plot the motion timeline
fig, ax = plt.subplots()
ax.plot(timeline, motion)
ax.set(xlabel='Time (ms)', ylabel='Global Motion Estimation (pixel)', title='Global Motion Estimation over Time')
ax.grid()
fig.savefig(image_folder+"GME.png")
plt.show()


