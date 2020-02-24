from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time as T
import cv2
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
event = False
time_t = 0

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

frame = vs.read()[1]
frame_t = None
# loop over the frames of the video
while True:
        time = count/fps
        grabbed = vs.grab()
        if(grabbed):
            count += 1
            if((time - time_t) >= 1):
                #grab the current frame
                frame_t = vs.retrieve()[1]
                time_t = time
            else:
                continue
        else:
            break

        # resize the frame, convert it to grayscale, and blur it
        #frame = imutils.resize(frame, width=800)
        #frame = apply_mask(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        gray_t = cv2.GaussianBlur(gray_t, (21, 21), 0)
        
        
        # compute motion frame
        frameDelta = cv2.absdiff(gray, gray_t)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
    
        #Set by default value of event_in at False (if non contours)
        event_in = False
        # loop over the contours
        for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < args["min_area"]:
                        continue
                
                event_in = True
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Occupied"

        #Print the timeline when the status of the field change
        if(event ^ event_in):
                time_in = count/fps
                timeline.append([time, time_in, event])
                event = event_in
                time = time_in


        # draw the text and timestamp on the frame if display is set at false
        if (args["display"]):

            cv2.putText(frame, "Field status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

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
        frame = frame_t

# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
timeline.append([time, count/fps, event])
timeline = timeline[1:]
print(timeline)

print("Execution time : "+str(T.time() - start_time))
