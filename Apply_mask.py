import cv2
import argparse
import numpy as np

mask = np.zeros((720, 1280, 3))

for row in range(mask.shape[0]):
    for column in range(mask.shape[1]):
        if(row >= 580):
            mask[row, column] = 1

        if( (0 <= row <= 90) & ( 840 <= column <= 1200)):
            mask[row, column] = 1

        if( (250 <= row <= 600) & ( 1240 <= column <= 1280)):
            mask[row, column] = 1


def apply_mask(image):
    result = image.copy()
    result[mask!=0] = 0
    return(result)


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", help = "path to the image on which the mask is applying")

    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    result = image.copy()

    print(image.shape)

    [rows, columns, channels] = image.shape

    mask = np.zeros((rows,columns,channels))

    for row in range(rows):
            for column in range(columns):
                if(row >= 580):
                    mask[row, column] = 1

                if( (0 <= row <= 90) & ( 840 <= column <= 1200)):
                    mask[row, column] = 1

    result[mask!=0] = 0

    cv2.imshow("Masked image", result)
    plt.show()
    cv2.waitKey()
