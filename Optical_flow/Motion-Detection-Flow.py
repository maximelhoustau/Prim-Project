from imutils.video import VideoStream
import argparse
import imutils
import time as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.Apply_mask import apply_mask
from scipy import signal

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-d", "--display", type=int, default=1, help="minimum area size")
ap.add_argument("-r", "--resized", type=int, default=0, help="Set the width to resize the image")
ap.add_argument("-m", "--masked", type=int, default=0, help="Set whether to mask the frames or not")
ap.add_argument("-t", "--timed", type=float, default=1, help="Set the time between 2 frames to analyse")

args = vars(ap.parse_args())

displayed = args["display"]
video_title = args["video"]
resized = args["resized"]
masked = args["masked"]
timed = args["timed"]

image_folder = "./images/"
video_folder = "./videos/"

print("Video: "+video_folder+video_title)
print("Displayed : "+str(displayed))

def MDF(masked, resized):
    start_time = T.time()
    timeline = []
    motion = []
    time_t = 0

    # if the video argument is None, then we are reading from webcam
    if args.get("video", None) is None:
        vs = VideoStream(src=0).start()
        time.sleep(2.0)

    # otherwise, we are reading from a video file
    else:
        vs = cv2.VideoCapture(video_folder + video_title)

    # Get the fps and number of frame to get the time of the video
    fps = vs.get(cv2.CAP_PROP_FPS)
    count = 0

    # loop over the frames of the video
    frame = vs.read()[1]
    frame_t = None
    while True:
        time = count/fps
        grabbed = vs.grab()
        if(grabbed):          
            count += 1
            if((time - time_t) >= timed):
                # grab the current frame 
                frame_t = vs.retrieve()[1] 
                time_t = time
            else:
                continue
        else:
            break
        
        #Apply masked or not
        if(masked): 
            frame = apply_mask(frame) 
            frame_t = apply_mask(frame_t) 
        
        #Apply resized or not
        if(resized):
            frame = imutils.resize(frame, width=resized)
            frame_t = imutils.resize(frame_t, width=resized)       


        # convert it to grayscale, and blur it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        gray_t = cv2.GaussianBlur(gray_t, (21, 21), 0)
        
        # compute motion frames
        frameDelta = cv2.absdiff(gray, gray_t)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        #thresh = cv2.dilate(thresh, None, iterations=2)
    
        #Show the real time videos if display is not set at 0
        if (displayed):

            # show the frame and record if the user presses a key
            cv2.imshow("Frame", frame)
            cv2.imshow("Thresh", thresh)
            cv2.imshow("Frame Delta", frameDelta)

        # if the `q` key is pressed, break from the lop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
                break
            
        if key == ord("w"):
            cv2.imwrite(image_folder+"gray_n.jpg", gray)
            cv2.imwrite(image_folder+"gray_n+1.jpg", gray_t)
            cv2.imwrite(image_folder+"MD_thresh.jpg", thresh)
            cv2.imwrite(image_folder+"MD_delta.jpg", frameDelta)
        
        #Add time and motion counter of this frame to their vector in order to plot
        timeline.append(count/fps)  
        motion.append(np.count_nonzero(thresh))
        #update frame
        frame = frame_t

    # cleanup the camera and close any open windows
    vs.stop() if args.get("video", None) is None else vs.release()
    cv2.destroyAllWindows()
    
    print("\nMasked : "+str(masked))
    print("Size of the frames : "+str(frame.shape))
    print("Execution time : "+str(T.time() - start_time))
    print(timeline)
    return(timeline, motion)

timeline, motion = MDF(masked = masked, resized = resized)

plt.plot(timeline, motion)
plt.plot([210,210] , [0, max(motion)])
plt.plot([1570,1570] , [0, max(motion)])
plt.plot([1780,1780] , [0, max(motion)])
plt.plot([3260,3260] , [0, max(motion)])
plt.show()

b, a = signal.butter(3, 0.05)
#zi = signal.lfilter_zi(b, a)
#z, _ = signal.lfilter(b, a, motion, zi=zi*motion[0])
#z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
#y = signal.filtfilt(b, a, motion)
y = signal.filtfilt(b, a, motion)

plt.plot(timeline, y)
plt.plot([210,210] , [0, max(motion)])
plt.plot([1570,1570] , [0, max(motion)])
plt.plot([1780,1780] , [0, max(motion)])
plt.plot([3260,3260] , [0, max(motion)])
plt.show()


