import cv2
import numpy as np
import time
from picamera2 import Picamera2
import libcamera
from LOBOROBOT2 import LOBOROBOT, FORWARD

# ##################################################################
# --- 1. 關鍵參數調整區 ---
# ##################################################################

# --- 馬達控制 ---
BASE_SPEED = 30  

# --- PD 控制器增益 (Gains) ---
Kp = 0.15 
Kd = 0.08

# --- 影像處理 ---
WW, HH = 320, 240 # 攝影機影像寬高
ROI_Y_START = int(HH * 0.75) # 從 3/4 高度開始
ROI_HEIGHT = 20              # 偵測區域的高度 (20 像素)

# --- 白色線條的 HSV 閥值 ---
LOWER_WHITE = np.array([0, 0, 185])
UPPER_WHITE = np.array([180, 50, 255])

# ##################################################################
# --- 2. 輔助函式 ---
# ##################################################################

def clamp(value, min_val, max_val):
    """將一個數值限制在指定的 min/max 範圍內"""
    return max(min_val, min(value, max_val))

# ##################################################################
# --- 3. 初始化 ---
# ##################################################################

print("正在初始化元件...")

# 初始化 LOBOROBOT
robot = LOBOROBOT()
robot.t_stop(0.1) # 確保馬達停止

# 初始化 Picamera2
picamera = Picamera2()
config = picamera.create_preview_configuration(
            main={"format": "RGB888", "size": (WW, HH)},
            transform=libcamera.Transform(hflip=1, vflip=1) 
)
picamera.configure(config)
picamera.start()
time.sleep(1.0) # 等待攝影機預熱
print("初始化完成。按下 Ctrl+C 停止程式。")

# ##################################################################
# --- 4. 主控制迴圈 ---
# ##################################################################

last_error = 0       # 上一次的誤差 (用於 D 項計算)
target_x = WW // 2   # 我們的目標 X 座標 (畫面正中央)

# --- [新邏輯] ---
# 用於追蹤丟失線條的時間戳記
line_lost_timestamp = None 
# 寬限期 (秒)
GRACE_PERIOD = 0.5 
# --- [新邏輯結束] ---

try:
    while True:
        # --- A. 影像擷取與處理 ---
        frame_rgb = picamera.capture_array()
        frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        mask_white = cv2.inRange(frame_hsv, LOWER_WHITE, UPPER_WHITE)
        roi = mask_white[ROI_Y_START : ROI_Y_START + ROI_HEIGHT, :]
        
        # --- B. 尋找車道中心 ---
        M = cv2.moments(roi)
        
        if M["m00"] > 0:
            # --- C. [線條已找到] ---
            
            # 1. 重置 "丟失線條" 計時器
            line_lost_timestamp = None 
            
            # 2. 計算質心 X 座標
            cX = int(M["m10"] / M["m00"])
            
            # 3. PD 控制器計算
            error = cX - target_x
            derivative = error - last_error
            turn_offset = (Kp * error) + (Kd * derivative)
            last_error = error
            
            # 4. 馬達控制
            left_speed = clamp(BASE_SPEED - turn_offset, 0, 100)
            right_speed = clamp(BASE_SPEED + turn_offset, 0, 100)

            robot.MotorRun(0, FORWARD, left_speed)
            robot.MotorRun(1, FORWARD, right_speed)
            robot.MotorRun(2, FORWARD, left_speed)
            robot.MotorRun(3, FORWARD, right_speed)

        else:
            # --- D. [未偵測到線條] (新的容錯邏輯) ---
            
            if line_lost_timestamp is None:
                # D1. 這是「剛剛」丟失線條的第一個影格：
                print(f"警告：未偵測到車道線！進入 {GRACE_PERIOD} 秒寬限期...")
                line_lost_timestamp = time.time() # 記錄當前時間
                last_error = 0 # 重置誤差
                
                # 保持直行
                robot.MotorRun(0, FORWARD, BASE_SPEED)
                robot.MotorRun(1, FORWARD, BASE_SPEED)
                robot.MotorRun(2, FORWARD, BASE_SPEED)
                robot.MotorRun(3, FORWARD, BASE_SPEED)

            else:
                # D2. 已經處於 "丟失" 狀態：
                elapsed_time = time.time() - line_lost_timestamp
                
                if elapsed_time < GRACE_PERIOD:
                    # 仍在 0.5 秒寬限期內
                    print(f"寬限期內，保持直行... {elapsed_time:.2f}s")
                    # 保持直行
                    robot.MotorRun(0, FORWARD, BASE_SPEED)
                    robot.MotorRun(1, FORWARD, BASE_SPEED)
                    robot.MotorRun(2, FORWARD, BASE_SPEED)
                    robot.MotorRun(3, FORWARD, BASE_SPEED)
                else:
                    # D3. 寬限期結束，仍然找不到線
                    print(f"{GRACE_PERIOD} 秒寬限期結束，仍未找到線條。停止。")
                    robot.t_stop(0)
                    # 保持停止狀態 (並稍作暫停以避免 CPU 佔用過高)
                    time.sleep(0.1) 


except KeyboardInterrupt:
    # 偵測到 Ctrl+C (手動控制停止)
    print("\n偵測到手動停止指令！")

finally:
    # --- 5. 清理 ---
    print("正在停止馬達並關閉攝影機...")
    robot.t_stop(1)
    picamera.stop()
    print("程式已終止。")
