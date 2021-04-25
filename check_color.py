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

cv2.createTrackbar('Hue_min','image',39,180,callback)
cv2.createTrackbar('Hue_max','image',95,180,callback)
cv2.createTrackbar('Sat_min','image',45,255,callback)
cv2.createTrackbar('Sat_max','image',111,255,callback)
cv2.createTrackbar('Val_min','image',153,255,callback)
cv2.createTrackbar('Val_max','image',255,255,callback)

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
    img_out_2 = img_out

    # 面積・重心計算付きのラベリング処理を行う
    num_labels, label_image, stats, center = cv2.connectedComponentsWithStats(img_mask)

    # 最大のラベルは画面全体を覆う黒なので不要．データを削除
    num_labels = num_labels - 1

    #for i in range(1, num_labels):
        #sizes = stats[1:, -1]
        # remove small object
        #if 20 < sizes[i - 1]:
            #stats[label_image == i] = 255

    if(num_labels > 5):
        num_labels = 5

    stats = np.delete(stats, 0, 0)
    center = np.delete(center, 0, 0)

    # 検出したラベルの数だけ繰り返す
    for index in range(num_labels):
        #ラベルのx,y,w,h,面積s,重心位置mx,myを取り出す
        x = stats[index][0]
        y = stats[index][1]
        w = stats[index][2]
        h = stats[index][3]
        s = stats[index][4]
        mx = int(center[index][0])
        my = int(center[index][1])
        print("(x,y)=%d,%d (w,h)=%d,%d s=%d (mx,my)=%d,%d"%(x, y, w, h, s, mx, my) )

        # ラベルを囲うバウンディングボックスを描画
        cv2.rectangle(img_out_2, (x, y), (x+w, y+h), (255, 0, 255))

        # 重心位置の座標と面積を表示
        cv2.putText(img_out_2, "%d,%d"%(mx,my), (x-15, y+h+15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))
        cv2.putText(img_out_2, "%d"%(s), (x, y+h+30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))

        # (X)ウィンドウに表示

    cv2.imshow('image',img_out)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break