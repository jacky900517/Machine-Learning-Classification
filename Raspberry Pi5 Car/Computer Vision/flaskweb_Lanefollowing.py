import cv2
import numpy as np
import time
import threading
import libcamera
from picamera2 import Picamera2
from flask import Flask, Response, render_template_string, jsonify
from LOBOROBOT2 import LOBOROBOT, FORWARD

# ##################################################################
# --- 1. 關鍵參數調整區 (來自 BasicRoadDetection.py) ---
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

# --- 循線邏輯寬限期 ---
GRACE_PERIOD = 0.5 

# ##################################################################
# --- 2. 網頁介面 (HTML) ---
# ##################################################################

INDEX_HTML = """
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>Raspberry Pi 5 循線車控制台</title>
  <style>
    body { font-family: Arial, sans-serif; text-align: center; background-color: #f0f0f0; }
    h3 { color: #333; }
    table { margin: 0 auto; border-collapse: collapse; }
    th, td { padding: 10px; border: 1px solid #ccc; background-color: #fff; }
    img { border-radius: 8px; border: 1px solid #333; display: block; }
    .controls { margin-top: 20px; }
    button {
      font-size: 16px;
      padding: 10px 20px;
      margin: 5px;
      border: none;
      border-radius: 5px;
      cursor: pointer;
    }
    #btn-start { background-color: #4CAF50; color: white; }
    #btn-stop { background-color: #f44336; color: white; }
    #status { margin-top: 10px; font-weight: bold; }
  </style>
</head>
<body>
  <h3>Raspberry Pi 5 循線車控制台</h3>
  
  <table>
    <tr>
      <th>原始影像</th>
      <th>車道線偵測 (處理後)</th>
    </tr>
    <tr>
      <td><img src="/live_original" alt="原始影像串流"></td>
      <td><img src="/live_processed" alt="處理後影像串流"></td>
    </tr>
  </table>

  <div class="controls">
    <button id="btn-start" onclick="startExecution()">開始執行循線</button>
    <button id="btn-stop" onclick="stopExecution()">結束執行循線</button>
    <div id="status">狀態：已停止</div>
  </div>

  <script>
    async function startExecution() {
      try {
        const response = await fetch('/api/start_execution', { method: 'POST' });
        const data = await response.json();
        if (data.ok) {
          document.getElementById('status').innerText = '狀態：正在執行循線...';
        }
      } catch (e) {
        console.error('Error starting:', e);
      }
    }

    async function stopExecution() {
      try {
        const response = await fetch('/api/stop_execution', { method: 'POST' });
        const data = await response.json();
        if (data.ok) {
          document.getElementById('status').innerText = '狀態：已停止';
        }
      } catch (e) {
        console.error('Error stopping:', e);
      }
    }
  </script>
</body>
</html>
"""

# ##################################################################
# --- 3. 全域變數與執行緒鎖 ---
# ##################################################################

# --- 執行緒控制 ---
running = True              # 控制所有背景執行緒是否繼續
execution_running = False   # 控制「循線邏輯」是否啟動
latest_frame = None         # 儲存最新的「原始」影像
latest_processed_frame = None # 儲存最新的「處理後」影像

# --- 執行緒鎖 (Locks) ---
frame_lock = threading.Lock()
processed_frame_lock = threading.Lock()
execution_lock = threading.Lock()

# --- PD 控制器變數 ---
last_error = 0
line_lost_timestamp = None
target_x = WW // 2

# --- 輔助函式 ---
def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

# ##################################################################
# --- 4. 初始化 ---
# ##################################################################

print("正在初始化元件...")

# --- 初始化 LOBOROBOT ---
robot = LOBOROBOT()
robot.t_stop(0.1)

# --- 初始化 Picamera2 ---
picamera = Picamera2()
config = picamera.create_preview_configuration(
            main={"format": "RGB888", "size": (WW, HH)},
            transform=libcamera.Transform(hflip=1, vflip=1) 
)
picamera.configure(config)
picamera.start()
time.sleep(1.0) # 等待攝影機預熱

# --- 初始化「處理後影像」為黑色畫面 ---
blank_frame = np.zeros((HH, WW, 3), dtype=np.uint8)
cv2.putText(blank_frame, "Waiting for data...", (WW//2 - 70, HH//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
latest_processed_frame = blank_frame.copy()

print("初始化完成。")

# ##################################################################
# --- 5. 背景執行緒 (Threads) ---
# ##################################################################

# --- 執行緒 1: 攝影機擷取 ---
def capture_loop():
    """
    僅負責從攝影機擷取畫面並存到 global `latest_frame`
    """
    global latest_frame, running
    
    while running:
        frame_rgb = picamera.capture_array()
        with frame_lock:
            latest_frame = frame_rgb
        time.sleep(0.01) # 稍微讓出 CPU，避免 100% 佔用
    print("攝影機擷取執行緒已停止。")

# --- 執行緒 2: 循線與馬達控制 ---
def motor_control_loop():
    """
    負責處理影像、計算PD、控制馬達
    """
    global running, execution_running, latest_frame, latest_processed_frame
    global last_error, line_lost_timestamp, target_x

    while running:
        # 1. 取得目前是否該「執行」的狀態
        with execution_lock:
            is_active = execution_running
            
        # 2. 取得最新畫面
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame_rgb = latest_frame.copy()

        # 3. --- 影像處理 (無論是否啟動都執行，以提供串流) ---
        frame_hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
        mask_white = cv2.inRange(frame_hsv, LOWER_WHITE, UPPER_WHITE)
        roi = mask_white[ROI_Y_START : ROI_Y_START + ROI_HEIGHT, :]
        
        # 將處理後的 mask 轉換為 BGR (3通道) 以便串流
        processed_img_for_stream = cv2.cvtColor(mask_white, cv2.COLOR_GRAY2BGR)
        
        # 儲存處理後的影像供串流
        with processed_frame_lock:
            latest_processed_frame = processed_img_for_stream

        # 4. --- 馬達控制 (僅在 `is_active` 為 True 時執行) ---
        if not is_active:
            # 如果是剛被切換為 "停止"
            if last_error != 0 or line_lost_timestamp is not None:
                print("執行已停止，正在停止馬達。")
                robot.t_stop(0)
                last_error = 0
                line_lost_timestamp = None
            time.sleep(0.1) # 停止狀態下，降低迴圈頻率
            continue

        # --- 以下為 is_active == True 的循線邏輯 ---
        M = cv2.moments(roi)
        
        if M["m00"] > 0:
            # C. [線條已找到]
            line_lost_timestamp = None 
            cX = int(M["m10"] / M["m00"])
            
            error = cX - target_x
            derivative = error - last_error
            turn_offset = (Kp * error) + (Kd * derivative)
            last_error = error
            
            left_speed = clamp(BASE_SPEED - turn_offset, 0, 100)
            right_speed = clamp(BASE_SPEED + turn_offset, 0, 100)

            robot.MotorRun(0, FORWARD, left_speed)
            robot.MotorRun(1, FORWARD, right_speed)
            robot.MotorRun(2, FORWARD, left_speed)
            robot.MotorRun(3, FORWARD, right_speed)

        else:
            # D. [未偵測到線條]
            if line_lost_timestamp is None:
                print(f"警告：未偵測到車道線！進入 {GRACE_PERIOD} 秒寬限期...")
                line_lost_timestamp = time.time()
                last_error = 0 
                robot.MotorRun(0, FORWARD, BASE_SPEED)
                robot.MotorRun(1, FORWARD, BASE_SPEED)
                robot.MotorRun(2, FORWARD, BASE_SPEED)
                robot.MotorRun(3, FORWARD, BASE_SPEED)
            else:
                elapsed_time = time.time() - line_lost_timestamp
                if elapsed_time < GRACE_PERIOD:
                    print(f"寬限期內，保持直行... {elapsed_time:.2f}s")
                    robot.MotorRun(0, FORWARD, BASE_SPEED)
                    robot.MotorRun(1, FORWARD, BASE_SPEED)
                    robot.MotorRun(2, FORWARD, BASE_SPEED)
                    robot.MotorRun(3, FORWARD, BASE_SPEED)
                else:
                    print(f"{GRACE_PERIOD} 秒寬限期結束，停止。")
                    robot.t_stop(0)
                    time.sleep(0.1) # 停止後稍作等待

        time.sleep(0.01) # 循線迴圈的延遲

    # 迴圈結束 (running = False)
    robot.t_stop(0.1)
    print("馬達控制執行緒已停止。")


# ##################################################################
# --- 6. Flask 網頁伺服器 ---
# ##################################################################

app = Flask(__name__)

def mjpeg_generator(source_type):
    """
    影像串流產生器，根據 source_type 決定串流來源
    """
    global latest_frame, latest_processed_frame
    
    while True:
        frame = None
        
        if source_type == 'original':
            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()
            
            if frame is None:
                time.sleep(0.01)
                continue
                
            # 將 RGB 轉為 BGR 供 cv2 編碼
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        elif source_type == 'processed':
            with processed_frame_lock:
                if latest_processed_frame is not None:
                    frame = latest_processed_frame.copy()
            
            if frame is None:
                time.sleep(0.01)
                continue
            
            # 已經是 BGR 格式，不需轉換
            frame_bgr = frame
        
        # 編碼為 JPEG
        ok, jpeg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            continue
            
        # 產生 MJPEG 串流
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() +
               b"\r\n")

@app.route("/")
def index():
    """主頁面"""
    return render_template_string(INDEX_HTML)

@app.route("/live_original")
def video_feed_original():
    """原始影像串流"""
    return Response(mjpeg_generator(source_type='original'),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/live_processed")
def video_feed_processed():
    """處理後影像串流"""
    return Response(mjpeg_generator(source_type='processed'),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/start_execution", methods=["POST"])
def start_execution():
    """API: 開始循線"""
    global execution_running
    with execution_lock:
        execution_running = True
    print("API: 收到開始執行指令。")
    return jsonify({"ok": True, "status": "started"})

@app.route("/api/stop_execution", methods=["POST"])
def stop_execution():
    """API: 停止循線"""
    global execution_running
    with execution_lock:
        execution_running = False
    print("API: 收到停止執行指令。")
    return jsonify({"ok": True, "status": "stopped"})

def cleanup():
    """程式結束前的清理工作"""
    global running
    print("\n正在關閉程式...")
    running = False # 通知所有執行緒停止
    
    try:
        t_camera.join(timeout=1.0)
    except Exception as e:
        print(f"關閉攝影機執行緒時發生錯誤: {e}")
        
    try:
        t_motor.join(timeout=1.0)
    except Exception as e:
        print(f"關閉馬達執行緒時發生錯誤: {e}")
        
    robot.t_stop(0.5)
    picamera.stop()
    print("清理完畢，程式已終止。")

# ##################################################################
# --- 7. 主程式啟動 ---
# ##################################################################

if __name__ == "__main__":
    
    # 啟動背景執行緒
    t_camera = threading.Thread(target=capture_loop, daemon=True)
    t_motor = threading.Thread(target=motor_control_loop, daemon=True)
    
    t_camera.start()
    t_motor.start()
    
    print("啟動 Flask 伺服器，請在瀏覽器開啟 http://[您的Pi的IP]:5000")
    
    try:
        # 啟動 Flask 伺服器
        # threaded=True 確保 Flask 可以同時處理多個請求 (例如 API 和 2個影像串流)
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    
    except KeyboardInterrupt:
        print("\n偵測到 Ctrl+C")
    
    finally:
        # 執行清理
        cleanup()
