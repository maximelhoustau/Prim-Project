# AI project on football analysis

The goal of this project was to detect key moments of amateur soccer videos. In particular, we try to predict the **timeline** : when the match starts, pauses, starts again and then ends.

To achieve it, I performed some **group activity recognition** based on pose extraction by the deep learning model **Openpose**. Machine Learning tools like **preprocessing**, **clustering** and **SVM** were used.

Here is an extract of video with pose points detected by Openpose, that I worked with : 

![Video Sample](./videos/sample.gif)

For example this extract of video contains first, the warm-up (match did not start), and then the match starts. I have to make the distinction between players playing football during warm-up and players playing football in a match.

To summarize, a clustering is used on poses from every frame of the video in order to compute the 10 average zones where the players are located when they are on the field. Global movement of every detected player is computed from one frame to another. Several other features are computed before feeding it to a binary classifier. Finally, we predict the match status base on **2 successive frames**.

I am finally able to predict the timeline of any video with **100% accuracy**. After the first results, a real-time detection can be cansiderer. Transfer learning and global model can also be considered after those promising results.
