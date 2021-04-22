import numpy as np
import cv2
import argparse

# CLI: for parse args
parser = argparse.ArgumentParser(description="")
parser.add_argument('target', help='Target image')
args = parser.parse_args()

def callback():
    return 0

img = cv2.imread(args.target)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.namedWindow('image')

cv2.createTrackbar('Hue_min','image',0,180,callback)
cv2.createTrackbar('Hue_max','image',0,180,callback)
cv2.createTrackbar('Sat_min','image',0,255,callback)
cv2.createTrackbar('Sat_max','image',0,255,callback)
cv2.createTrackbar('Val_min','image',0,255,callback)
cv2.createTrackbar('Val_max','image',0,255,callback)

while(1):
    hue_min = cv2.getTrackbarPos('Hue_min','image')
    hue_max = cv2.getTrackbarPos('Hue_max','image')
    sat_min = cv2.getTrackbarPos('Sat_min','image')
    sat_max = cv2.getTrackbarPos('Sat_max','image')
    val_min = cv2.getTrackbarPos('Val_min','image')
    val_max = cv2.getTrackbarPos('Val_max','image')

    hsv_low = np.array([hue_min,sat_min,val_min])
    hsv_upper = np.array([hue_max,sat_max,val_max])

    img_mask = cv2.inRange(img_hsv, hsv_low, hsv_upper)
    img_out = cv2.bitwise_and(img, img, mask=img_mask)

    cv2.imshow('image',img_out)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break