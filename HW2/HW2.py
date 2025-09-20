import cv2
from datetime import datetime
import time

# 擷取我的視訊鏡頭
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Cannot open camera"); raise SystemExit

#opencv預設擷取到的不是1080p 60FPS 我在這邊特別要求set
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

#讀取我實際抓取到的解析度&幀數
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_wr = cap.get(cv2.CAP_PROP_FPS)

#設定影片輸出的編碼
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#設定影片輸出格式跟名稱
out = cv2.VideoWriter('myFPS&TIME.mp4', fourcc, fps_wr, (width, height))

if not out.isOpened():
    print("VideoWriter not opened"); raise SystemExit

# --- 文字樣式 --- by AI code
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 0.7
thick = 2
fg = (255, 255, 255)   # 白
bg = (0, 0, 0)         # 黑（描邊）
margin = 10

#FPS 計算
prev_t = time.time() #上一幀的時間
fps_ema = 0.0
alpha = 0.1  

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame"); break

    # 更新 FPS
    now = time.time()   #現在時間
    dt = now - prev_t   #兩幀的時間差
    prev_t = now        #更新時間
    inst_fps = 1.0 / dt if dt > 0 else 0.0  #即時FPS
    fps_ema = inst_fps if fps_ema == 0 else (1-alpha)*fps_ema + alpha*inst_fps #平滑處理

######################################################
######################################################
    #文字部分by AI code
    # 準備文字
    left_text  = f"FPS = {fps_ema:5.1f}"
    right_text = datetime.now().strftime("%Y/%m/%d %H:%M:%S")

    # 左上：先描邊再實字
    (lw, lh), _ = cv2.getTextSize(left_text, font, scale, thick)
    lx, ly = margin, margin + lh
    cv2.putText(frame, left_text, (lx+1, ly+1), font, scale, bg, thick+2, cv2.LINE_AA)
    cv2.putText(frame, left_text, (lx,   ly),   font, scale, fg, thick,   cv2.LINE_AA)

    # 右上：靠右對齊
    (rw, rh), _ = cv2.getTextSize(right_text, font, scale, thick)
    rx, ry = frame.shape[1] - rw - margin, margin + rh
    cv2.putText(frame, right_text, (rx+1, ry+1), font, scale, bg, thick+2, cv2.LINE_AA)
    cv2.putText(frame, right_text, (rx,   ry),   font, scale, fg, thick,   cv2.LINE_AA)
    
######################################################
######################################################
    
    out.write(frame) #把文字寫進影片
    cv2.imshow('oxxostudio', frame) #顯示視窗
    if cv2.waitKey(1) == ord('q'): #設定按q離開
        break

cap.release()
out.release()
cv2.destroyAllWindows()

