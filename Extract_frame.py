import cv2

vidcap = cv2.VideoCapture('./videos/test.mp4')
time = 1000*6 + 27*60*1000
vidcap.set(cv2.CAP_PROP_POS_MSEC,time)      # just cue to 20 sec. position
success,image = vidcap.read()
if success:
    cv2.imwrite("./images/fieldref.jpg", image)     # save frame as JPEG file
    cv2.imshow(str(time)+" sec",image)
    cv2.waitKey()
