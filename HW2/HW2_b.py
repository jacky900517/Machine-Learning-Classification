import cv2

def make_tracker():
    try:
        return cv2.legacy.TrackerMOSSE_create()
    except AttributeError:
        return cv2.TrackerMOSSE_create()

tracker_list = []
colors = [(0,0,255),(0,255,255),(255,255,0)]
tracking = False

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

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
out = cv2.VideoWriter('HW2_b.mp4', fourcc, fps_wr, (width, height))

if not cap.isOpened():
    print("Cannot open camera"); exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame"); break

    # 追蹤並畫框
    if tracking:
        for i, t in enumerate(tracker_list):
            ok, box = t.update(frame)
            if ok:
                x, y, w, h = map(int, box)
                cv2.rectangle(frame, (x, y), (x+w, y+h), colors[i % 3], 3)

    cv2.imshow('oxxostudio', frame)
    keyName = cv2.waitKey(1) & 0xFF

    if keyName == ord('q'):
        break

    if keyName == ord('a') and not tracking:
        tracker_list = []
        for _ in range(3):
            area = cv2.selectROI('oxxostudio', frame, showCrosshair=False, fromCenter=False)
            if area == (0,0,0,0):  # 取消就跳過
                continue
            t = make_tracker()
            t.init(frame, area)
            tracker_list.append(t)
        tracking = len(tracker_list) > 0
    
    out.write(frame) 
    
out.release()
cap.release()
cv2.destroyAllWindows()
