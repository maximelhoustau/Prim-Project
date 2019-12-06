# Prim-Project

This is the repository that contains all code written during a project I worked on with the French start-up Footbar. The goal of this AI project is to analyse videos of 5 vs 5 football matches in order to detect specific actions over time and measure some physical caracteristics of the players on the field (speed for example).

* __Motion-Detection-Flow.py__

This file computes the absolute difference between successful frames of a video, and then plot the number of pixels activated by the difference followed by a threshold as a measure of global mouvement estimation. We then plot and save it. Here is how to use it:
```bash
python3 Motion-Detection-Flow.py -v 'path_to_video' -i 'path_to_initalization_image' -d 'display'
```
>* -v : [str] option to pass the path of the video we want as input. Without it, it will get the video stream of your laptop's webcam:
>* -d : [int] choose if you want the real time display or not. Default = 1 else 0
>* -r : [int] size of the width to resize the video frames. Default = 0

Here are the outputs with the following terminal line:
```bash
python3 Motion-Detection-Flow.py -v timeline_test.mp4 -r 500
```
We have the comparisons of the signals in output of the algorithm when we apply a mask or not, and when we resize the image to a width of 500:
![GMEComp500](/images/GME_comparison_500.png)

The execution time comparison is made in the terminal output:
```bash
Video: ./videos/timeline_test.mp4
Displayed : 0

Masked : 0
Size of the frames : (720, 1280, 3)
Execution time : 55.12852096557617

Masked : 1
Size of the frames : (720, 1280, 3)
Execution time : 97.16807866096497

Masked : 0
Size of the frames : (281, 500, 3)
Execution time : 49.86319613456726

Masked : 1
Size of the frames : (281, 500, 3)
Execution time : 83.85430908203125

```

Here are the outputs with the following terminal line:
```bash
python3 Motion-Detection-Flow.py -v timeline_test.mp4 -r 1000
```
We have the comparisons of the signals in output of the algorithm when we apply a mask or not, and when we resize the image to a width of 1000:
![GMEComp1000](/images/GME_comparison_1000.png)

The execution time comparison is made in the terminal output:
```bash
Video: ./videos/timeline_test.mp4
Displayed : 0

Masked : 0
Size of the frames : (720, 1280, 3)
Execution time : 56.564244985580444

Masked : 1
Size of the frames : (720, 1280, 3)
Execution time : 94.05397605895996

Masked : 0
Size of the frames : (281, 500, 3)
Execution time : 49.49067497253418

Masked : 1
Size of the frames : (281, 500, 3)
Execution time : 81.78049516677856

```

* __Motion-Detection-Ref.py__

This file basically takes a video and shows the objects moving into it. We suppose here that the camera is fixed and that the background is always the same. The principle of the algorithm is to store a frame containing the background and compute the difference between it and every other frames of the video to show what changed. Here is how to use it : 
```bash
python3 Motion-Detection-Ref.py -v 'path_to_video' -a 'size_of_minimal_detected_box' -i 'path_to_initalization_image' -d 'display'
```
>* -v : [str] option to pass the path of the video we want as input. Without it, it will get the video stream of your laptop's webcam
>* -a : [int] size of the minimal box to detect to avoid noise detection. Default = 500
>* -i : [str] option to pass the path of the image we would like to initialize the background; otherwise the background will be the first frame of the video/webcam. Be careful, the initialization image need to have the same size as the frames of the videos.
>* -d : [int] choose if you want the real time display or not. Default = 1 else 0

Here are the outputs with the following terminal line:
```bash
python3 Motion-Detection-Ref.py -v timeline_test.mp4 -i timeline_test_ref.jpg -a 500
```
We can see in black the mask applied to avoid unwanted detection in these particular zones:
![FrameDetection](/images/MD_frame.jpg)
![DeltaFrame](/images/MD_delta.jpg)
![ThreshFrame](/images/MD_thresh.jpg)
![RefFrame](/images/fieldref.jpg)

The terminal output is the following timeline:
```bash
[[0.0, 48.6, True], [48.6, 104.8, False], [104.8, 180.04, True]]
```
The two first element of each vector are the beginning and the ending time of this interval, the third element is a boolean which indicates if there are mouvements on the fields (i.e if there is someone playing).

* __Segmentation.py__

This file takes as input the five_test.png picture located into the images folder. Here we test 3 techniques of segmentation on the input picture: mean segmentation, instance segmentation and a k-mean segmentation. Just run the fowwing to obtain the computed images:
```bash
python3 Segmentation.py
```
![GrayedOriginal](/images/initial_image.jpg "Original grayed frame of the video")
![MeanSeg](/images/mean_seg_gray.jpg "Mean Segmentation on the original frame")
![InstSeg](/images/instance_seg_gray.jpg "Instance Segmentation on the original frame")
![KMean](/images/kmean_seg.jpg "K-Mean Segmentation on the original frame")

* __Seg_RCNN.py__

*tensorflow version < 2 required*

This file runs the pre-trained Mask R-CNN model (Folder Mask_RCNN) and performs object segmentation as well as object detection. To run it on the test image, just run the following:
```bash
python3 Seg_RCNN.py
```
Here is the result of the segmentation and detection performed on the R-CNN:
![R-CNN](/images/R-CNN_seg.png "Segmentation and object detection by R-CNN")
We have also access to the masks computed by the R-CNN:
![R-CNNMask](/images/mask0.png "Example of a mask computed by R-CNN")


