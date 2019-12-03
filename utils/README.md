# Utils folder

This folder contains several useful file for this project

* __Apply_mask.py__
This file apply a personal mask to the frames of a video in order to mask the unwanted detection zone into videos. A fonction can be used to apply this mask to any image. If used as main:
```bash
python3 Apply_mask.py -i 'path_to_image'
```
>* -i : [str] Name of the image file into ./images folder

It will display the input image with the mask applied. 

* __Extract_frame.py__
This file extract a frame from the videos folder into a picture in the images folder at the minute specified on the following command line: 
```bash
python3 Extract_frame.py -v 'path_to_video' -i 'path_to_image' -t time
```
>* -v : [str] Name of the video file into ./videos folder
>* -i : [str] Name of the image file into ./images folder
>* -t : [int] Time in second of the video frame to save

* __Extract_video.py__
This file extract a video from the videos folder into another shorter video of the same folder. The starting time to cut and the ended time to cut are indicated into the command line options that follow: 
```bash
python3 Extract_video.py -v 'path_to_original_video' -d 'path_to_saved_video' -s start_time -e end_time
```
>* -v : [str] Name of the video file into ./videos folder to cut
>* -d : [str] Name of the video file to save into ./videos folder
>* -s : [int] Starting time in minute to cut
>* -e : [int] Ending time in minute to cut
