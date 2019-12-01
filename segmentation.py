from skimage.color import rgb2gray
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from scipy import ndimage
from sklearn.cluster import KMeans

def mean_seg(image):
    gray = rgb2gray(image)
    gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
    for i in range(gray_r.shape[0]):
        if gray_r[i] > gray_r.mean():
            gray_r[i] = 1
        else:
            gray_r[i] = 0
    gray = gray_r.reshape(gray.shape[0],gray.shape[1])
    cv2.imshow("Mean segmentation", gray)

def instance_seg(image):
    gray = rgb2gray(image)
    gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
    print(gray_r.mean())
    print(gray_r.max())
    print(gray_r.min())
    for i in range(gray_r.shape[0]):
        if gray_r[i] > 0.7:
            gray_r[i] = 3
        elif gray_r[i] > 0.6:
            gray_r[i] = 2
        elif gray_r[i] > 0.4:
            gray_r[i] = 1
        else:
            gray_r[i] = 0
    gray = gray_r.reshape(gray.shape[0],gray.shape[1])
    #cv2.imshow("Instance segmentation", gray)
    plt.imshow(gray, cmap='gray')
    plt.show()

def k_mean_seg(image, nb_cluster):
    pic = image/255  # dividing by 255 to bring the pixel values between 0 and 1
    #cv2.imshow("Kmean initial", pic)
    pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])
    kmeans = KMeans(n_clusters=nb_cluster, random_state=0).fit(pic_n)
    pic2show = kmeans.cluster_centers_[kmeans.labels_]
    cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
    cv2.imshow("Kmean with "+str(nb_cluster)+" clusters", cluster_pic)


if __name__ == "__main__":

    image = cv2.imread('./images/five_test.png')
    image = imutils.resize(image, width = 500)
    
    k_mean_seg(image, 12)
    
    gray = rgb2gray(image)
    print(gray.shape)
    cv2.imshow("Field", gray)
    mean_seg(image)
    instance_seg(image)

    while True:
        key = cv2.waitKey(1) &0xFF
        if key == ord("q"):
            break

