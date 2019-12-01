# Footbar-Project

This is the repository that contains all code written during a project I worked on with the French start-up Footbar. The goal of this AI project is to analyse videos of 5 vs 5 football matches in order to detect specific actions over time and measure some physical caracteristics of the players on the field (speed for example).

* __Motion-Detection.py__

This file basically takes a video and shows the objects moving into it. We suppose here that the camera is fixed and that the background is always the same. The principle of the algorithm is to store a frame containing the background and compute the difference between it and every other frames of the video to show what changed. Here is how to use it : 
```bash
python3 Motion-Detection.py -v 'path_to_video' -a 'size_of_minimal_detected_box' -i 'path_to_initalization_image'
```
>> -v : [str] option to pass the path of the video we want as input. Without it, it will get the video stream of your laptop's webcam
>> -a : [int] size of the minimal box to detect to avoid noise detection. Default = 500
>> -i : [str] option to pass the path of the image we would like to initialize the background; otherwise the background will be the first frame of the video/webcam. Be careful, the initialization image need to have the same size as the frames of the videos.
