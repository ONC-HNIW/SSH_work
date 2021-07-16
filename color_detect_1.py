import numpy as np
import cv2
import argparse
import csv
import pprint

# CLI: for parse args
parser = argparse.ArgumentParser(description="")
parser.add_argument('target', help='Target video')
args = parser.parse_args()

video = cv2.VideoCapture(args.target)

while(video.isOpened()):
    ret, frame = video.read()

    points_1 = np.array([[1400,0],[1400,1080],[1920,1080],[1920,0]])
    cv2.fillPoly(frame, pts=[points_1], color=(0,0,0))

    points_2 = np.array([[0,0],[0,1080],[700,1080],[700,0]])
    cv2.fillPoly(frame, pts=[points_2], color=(0,0,0))
    
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    hsv_low = np.array([40,45,150])
    hsv_upper = np.array([85,120,255])
    frame_mask = cv2.inRange(frame_hsv, hsv_low, hsv_upper)
    frame_mask_filterd = cv2.medianBlur(frame_mask,55)
	
    frame_out = cv2.bitwise_and(frame, frame, mask=frame_mask)
    frame_out_2 = frame_out

    # 面積・重心計算付きのラベリング処理を行う
    num_labels, label_image, stats, center = cv2.connectedComponentsWithStats(frame_mask)

    # 最大のラベルは画面全体を覆う黒なので不要．データを削除
    num_labels = num_labels - 1

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
        cv2.rectangle(frame_out_2, (x, y), (x+w, y+h), (255, 0, 255))

        # 重心位置の座標と面積を表示
        cv2.putText(frame_out_2, "%d,%d"%(mx,my), (x-15, y+h+15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))
        cv2.putText(frame_out_2, "%d"%(s), (x, y+h+30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0))

        # (X)ウィンドウに表示
        cv2.imshow('OpenCV Window', frame_out_2)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
