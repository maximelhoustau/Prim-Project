from imutils.video import VideoStream
import argparse
import imutils
import time as T
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.Apply_mask import apply_mask

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-d", "--display", type=int, default=1, help="minimum area size")
#ap.add_argument("-r", "--resized", type=int, default=0, help="Set the width to resize the image")

args = vars(ap.parse_args())

displayed = args["display"]
video_title = args["video"]
#resized = args["resized"]
resized = [500, 1000]
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
            if((time - time_t) >= 0.07):
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
            cv2.imwrite(image_folder+"MD_frame.jpg", frame)
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
    
    return(timeline, motion)


fig, ax = plt.subplots(3, 2, sharex='col', sharey='row')

timeline1, motion1 = MDF(masked = 0, resized = 0)
ax[0,0].plot(timeline1, motion1)
ax[0,0].set(ylabel='GME (pixels)', title='Video Unmasked, Shape=(720,1280,3)')
ax[0,0].grid(True)

timeline2, motion2 = MDF(masked = 1, resized = 0)
ax[0,1].plot(timeline2, motion2)
ax[0,1].set(title = 'Video Masked, Shape=(720,1280,3)')
ax[0,1].grid(True)

timeline3, motion3 = MDF(masked = 0, resized = resized[1])
ax[1,0].plot(timeline3, motion3)
ax[1,0].set(ylabel='GME (pixels)', title = 'Video Unmasked, Shape=(666,1000,3')
ax[1,0].grid(True)

timeline4, motion4 = MDF(masked = 1, resized = resized[1])
ax[1,1].plot(timeline4, motion4)
ax[1,1].set(title = 'Video Masked, Shape=(666,1000,3)')
ax[1,1].grid(True)

timeline5, motion5 = MDF(masked = 0, resized = resized[0])
ax[2,0].plot(timeline5, motion5)
ax[2,0].set(xlabel='Time (ms)', ylabel='GME (pixels)', title = 'Video Masked, Shape=(281,500,3)')
ax[2,0].grid(True)

timeline6, motion6 = MDF(masked = 1, resized = resized[0])
ax[2,1].plot(timeline6, motion6)
ax[2,1].set(xlabel='Time (ms)', title = 'Video Masked, Shape=(281,500,3)')
ax[2,1].grid(True)

plt.show()
fig.savefig(image_folder+"GME_comparison.png")


fig, ax = plt.subplots(3, 1, sharex='col', sharey='row')
ax[0].plot(timeline1, [motion1[i] - motion2[i] for i in range(len(motion1))])
ax[0].set(ylabel='Noise (pixels)', title = 'Noise, Shape=(720,1080,3)')
ax[0].grid(True)

ax[1].plot(timeline3, [motion3[i] - motion4[i] for i in range(len(motion3))])
ax[1].set(ylabel='Noise (pixels)', title = 'Noise, Shape=(666,1000,3)')
ax[1].grid(True)

ax[2].plot(timeline5, [motion5[i] - motion6[i] for i in range(len(motion5))])
ax[2].set(xlabel = 'Time (ms)' ,ylabel='Noise (pixels)', title = 'Noise, Shape=(281,500,3)')
ax[2].grid(True)

plt.show()
fig.savefig(image_folder+"Noise_GME_comparison.png")











