# 匯入 OpenCV 函式庫，用於影像處理
import cv2
# 匯入 NumPy 函式庫，用於數值運算，特別是陣列操作
import numpy as np
# 匯入 math 函式庫，用於數學計算，如此處的平方根
import math 

# --- 全域變數 ---
# 定義一個類別來儲存跨幀的記憶和參數
class LaneMemory:
    # 初始化函數，設定所有記憶變數和參數的初始值
    def __init__(self):
        # 儲存左側車道線的擬合參數 (斜率, 截距)
        self.left_params = None
        # 儲存右側車道線的擬合參數 (斜率, 截距)
        self.right_params = None 
        # 儲存校準後的平均車道寬度 (在畫面底部)
        self.avg_width_bottom = None 
        # 儲存校準後的平均車道中心點 X 座標 (在畫面底部)
        self.avg_center_bottom = None 
        # 線對評分時，「寬度」差異的權重
        self.W_WIDTH = 1.0  
        # 線對評分時，「中心點」差異的權重
        self.W_CENTER = 1.5 
        # 訊號濾波器：允許線條底部 X 座標偏離預測位置的最大像素值
        self.POSITION_TOLERANCE_PIXELS = 60 
        # 訊號濾波器：線段長度低於此值被視為「弱訊號」
        self.STRONG_SIGNAL_LENGTH_THRESHOLD = 50 
        # 異常偵測：中心點單幀最大允許跳動像素 (相對於基準)
        self.MAX_CENTER_JUMP = 30 
        # 異常偵測：寬度單幀最大允許變化比例 (相對於基準)
        self.MAX_WIDTH_JUMP_RATIO = 0.15 
        # 恢復閾值：在異常狀態下，線對總分低於此值才能觸發恢復
        self.RECOVERY_THRESHOLD_SCORE = 50 
        # 標記當前是否處於「異常狀態」
        self.is_in_anomaly_state = False
        
        # --- 恢復冷卻 ---
        # 恢復成功後，禁用異常偵測的剩餘影格數
        self.anomaly_cooldown_frames = 0 
        # 恢復成功後，冷卻的總秒數
        self.RECOVERY_COOLDOWN_SECONDS = 3 
        
        # 校準階段的剩餘影格數
        self.calibration_frames = 10 
        # 平滑因子 (Exponential Moving Average 的權重)，值越小越平滑
        self.smoothing_factor = 0.05 
        # 儲存上一幀「良好」(非異常、非Coasting)狀態下的左線參數
        self.last_good_left_params = None
        # 儲存上一幀「良好」狀態下的右線參數
        self.last_good_right_params = None

# 建立 LaneMemory 類別的實例，用於儲存全域狀態
memory = LaneMemory()

# 根據線條的斜率和截距，計算用於繪圖的線段起點和終點座標
def make_coordinates(image, line_parameters):
    """ 計算座標 """
    # 嘗試解包線條參數，如果失敗(例如None)，使用預設值
    try:
        slope, intercept = line_parameters
    except (TypeError, ValueError): slope, intercept = 0.001, 0 
    # y1 設為影像底部
    y1 = image.shape[0] 
    # y2 設為影像下方約 85% 高度處 (決定綠色區域的遠近)
    y2 = int(y1 * 0.85) 
    # 防止斜率過小導致除以零或x值極大
    if abs(slope) < 0.001: slope = 0.001
    # 計算 x = (y - intercept) / slope
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    # 返回包含 [x1, y1, x2, y2] 的 NumPy 陣列
    return np.array([x1, y1, x2, y2])

# 計算給定線條參數在影像底部的 X 座標
def get_bottom_x(image, params):
    """ 計算底部 X """
    # 嘗試解包參數，處理可能的錯誤
    try:
        slope, intercept = params
    except (TypeError, ValueError): return None # 參數無效返回 None
    # 忽略斜率過小的線
    if abs(slope) < 0.001: return None
    # 獲取影像底部 y 座標
    y_bottom = image.shape[0]
    # 計算 x = (y - intercept) / slope
    x_bottom = (y_bottom - intercept) / slope
    # 檢查計算出的 x 是否在合理範圍內，防止極端值
    if -image.shape[1] < x_bottom < image.shape[1] * 2:
         return x_bottom # 返回計算出的 x 座標
    return None # 超出範圍返回 None

# 計算給定線段 (x1, y1, x2, y2) 的長度
def calculate_length(line_segment):
    """計算線段長度"""
    # 解包線段座標
    x1, y1, x2, y2 = line_segment
    # 使用畢氏定理計算長度
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# 主要的車道線偵測與追蹤函數
def find_and_track_lane_pair(image, lines, fps): 
    """
    *** 訊號濾波器 + 新版冷卻機制 ***
    """
    # 初始化候選列表
    all_left_candidates = [] # 儲存所有可能的左線 [( (斜率, 截距), (x1,y1,x2,y2) )]
    all_right_candidates = [] # 儲存所有可能的右線

    # 如果 HoughLinesP 回傳了線段
    if lines is not None:
        # 遍歷所有偵測到的線段
        for line in lines:
            # 獲取線段的端點座標
            x1, y1, x2, y2 = line.reshape(4)
            # 忽略垂直線 (避免 polyfit 除以零)
            if x1 == x2: continue
            # 計算線段的斜率和截距
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope, intercept = parameters[0], parameters[1]
            # 儲存線段原始座標
            segment = (x1, y1, x2, y2) 
            # 根據斜率範圍分類左右線
            if -1.8 < slope < -0.3: # 左線的合理斜率範圍
                all_left_candidates.append(((slope, intercept), segment))
            elif 0.3 < slope < 1.8: # 右線的合理斜率範圍
                all_right_candidates.append(((slope, intercept), segment))

    # --- *** 關鍵：訊號濾波器 *** ---
    # 初始化過濾後的候選列表
    filtered_left_candidates = [] # 只儲存 (slope, intercept)
    filtered_right_candidates = []

    # 預測當前幀線條應該在的底部 X 座標 (基於上一幀平滑後的記憶)
    predicted_lx_bottom = None
    predicted_rx_bottom = None
    if memory.left_params is not None: 
         predicted_lx_bottom = get_bottom_x(image, memory.left_params)
    if memory.right_params is not None:
         predicted_rx_bottom = get_bottom_x(image, memory.right_params)

    # 過濾左側候選線
    for params, segment in all_left_candidates:
        # 判斷是否為弱訊號 (長度太短)
        is_weak_signal = calculate_length(segment) < memory.STRONG_SIGNAL_LENGTH_THRESHOLD
        # 判斷是否超出預測邊界
        is_outside_boundary = False
        if predicted_lx_bottom is not None:
            current_lx_bottom = get_bottom_x(image, params)
            # 如果計算出的底部X有效 且 偏離預測位置過多
            if current_lx_bottom is not None and \
               abs(current_lx_bottom - predicted_lx_bottom) > memory.POSITION_TOLERANCE_PIXELS:
                is_outside_boundary = True
        # 核心濾波邏輯：如果 線條超出邊界 且 是弱訊號，則忽略
        if is_outside_boundary and is_weak_signal: continue 
        else: # 否則保留其參數
            filtered_left_candidates.append(params) 

    # 過濾右側候選線 (邏輯同左側)
    for params, segment in all_right_candidates:
        is_weak_signal = calculate_length(segment) < memory.STRONG_SIGNAL_LENGTH_THRESHOLD
        is_outside_boundary = False
        if predicted_rx_bottom is not None:
            current_rx_bottom = get_bottom_x(image, params)
            if current_rx_bottom is not None and \
               abs(current_rx_bottom - predicted_rx_bottom) > memory.POSITION_TOLERANCE_PIXELS:
                is_outside_boundary = True
        if is_outside_boundary and is_weak_signal: continue 
        else: filtered_right_candidates.append(params) 

    # --- 如果濾波後沒有任何左線或右線了 ---
    needs_coasting_due_to_detection_failure = False
    if not filtered_left_candidates or not filtered_right_candidates:
        needs_coasting_due_to_detection_failure = True # 標記需要 Coasting

    # --- 核心邏輯：評估過濾後的線對 ---
    best_pair_overall = None # 儲存當前幀分數最低的線對參數
    min_score_overall = float('inf') # 初始化最低分數為無限大
    potential_recovery_pairs = [] # 儲存異常狀態下，可能用於恢復的線對信息
    
    # 獲取畫面底部中心 X 座標
    screen_center_bottom = image.shape[1] / 2
    
    # 只有在濾波後還有線的情況下才進行評估
    if not needs_coasting_due_to_detection_failure:
        # 遍歷所有過濾後的左線
        for l_params in filtered_left_candidates: 
            # 遍歷所有過濾後的右線
            for r_params in filtered_right_candidates: 
                # 計算左右線在底部的 X 座標
                l_x_bottom = get_bottom_x(image, l_params)
                r_x_bottom = get_bottom_x(image, r_params)
                # 檢查座標是否有效，且左線在右線左邊
                if l_x_bottom is None or r_x_bottom is None or l_x_bottom >= r_x_bottom: continue 
                # 計算當前線對的寬度和中心點
                current_width = r_x_bottom - l_x_bottom
                current_center = (l_x_bottom + r_x_bottom) / 2
                
                # --- 校準階段 ---
                if memory.calibration_frames > 0:
                    # 選擇「跨越」畫面中心且「最窄」的線對作為初始基準
                    if l_x_bottom < screen_center_bottom and r_x_bottom > screen_center_bottom:
                        score = current_width # 用寬度作為分數
                        if score < min_score_overall:
                            min_score_overall = score
                            best_pair_overall = (l_params, r_params)
                # --- 追蹤階段 ---
                else:
                    # 計算寬度分數和中心點分數
                    score_width = abs(current_width - memory.avg_width_bottom) * memory.W_WIDTH
                    score_center = abs(current_center - memory.avg_center_bottom) * memory.W_CENTER
                    # 計算總分
                    total_score = score_width + score_center
                    # 如果當前線對總分更低，更新 overall best
                    if total_score < min_score_overall:
                        min_score_overall = total_score
                        best_pair_overall = (l_params, r_params)
                    # 如果處於異常狀態，檢查此線對是否可用於恢復
                    if memory.is_in_anomaly_state:
                        # 如果總分低於恢復閾值
                        if total_score < memory.RECOVERY_THRESHOLD_SCORE:
                             # 計算中心點差異
                             center_diff = abs(current_center - memory.avg_center_bottom)
                             # 記錄恢復候選信息 (總分, 中心點差異, 線對參數)
                             potential_recovery_pairs.append( (total_score, center_diff, (l_params, r_params)) )

    # --- 決定最終使用的線對 ---
    final_pair_to_use = None # 儲存最終決定使用的線對參數
    just_recovered = False # 標記是否剛從異常狀態恢復
    
    # 如果找到了恢復候選
    if potential_recovery_pairs: 
        # 按中心點差異排序，差異最小的優先
        potential_recovery_pairs.sort(key=lambda x: x[1]) 
        # 獲取最佳恢復線對的信息
        best_recovery_score, best_center_diff, best_recovery_pair_params = potential_recovery_pairs[0]
        # 設定最終使用的線對為恢復線對
        final_pair_to_use = best_recovery_pair_params
        # 解除異常狀態
        memory.is_in_anomaly_state = False 
        # 標記為剛恢復
        just_recovered = True
        # 啟動恢復冷卻計時器
        memory.anomaly_cooldown_frames = int(fps * memory.RECOVERY_COOLDOWN_SECONDS) 
        # 打印恢復信息
        print(f"  [恢復] CenterDiff: {best_center_diff:.0f}, Score: {best_recovery_score:.0f}. 啟動 {memory.RECOVERY_COOLDOWN_SECONDS} 秒冷卻。")
    # 如果仍在異常狀態 (沒找到恢復目標)
    elif memory.is_in_anomaly_state:
        # 使用當前幀評分最低的線對 (可能是錯誤的，例如閘道線)
        final_pair_to_use = best_pair_overall 
    # 如果是正常狀態
    else:
        # 使用當前幀評分最低的線對
        final_pair_to_use = best_pair_overall 

    # --- 異常偵測 ---
    anomaly_detected_this_frame = False # 標記當前幀是否偵測到異常
    force_coasting_in_cooldown = False # 標記是否因為冷卻期異常而強制Coasting

    # 只有在 找到了最終線對 且 校準已完成 時才進行異常偵測
    if final_pair_to_use is not None and memory.calibration_frames <= 0 :
        # 獲取最終線對的參數
        l_params, r_params = final_pair_to_use
        # 計算底部 X 座標
        l_x_bottom = get_bottom_x(image, l_params)
        r_x_bottom = get_bottom_x(image, r_params)
        # 確保計算成功
        if l_x_bottom is not None and r_x_bottom is not None: 
            # 計算當前中心點和寬度
            current_center = (l_x_bottom + r_x_bottom) / 2
            current_width = r_x_bottom - l_x_bottom
            # 計算與基準記憶的差異
            center_jump = abs(current_center - memory.avg_center_bottom)
            width_jump_ratio = abs(current_width - memory.avg_width_bottom) / memory.avg_width_bottom
            
            # 檢查是否超過異常閾值
            if center_jump > memory.MAX_CENTER_JUMP or width_jump_ratio > memory.MAX_WIDTH_JUMP_RATIO:
                # 標記偵測到異常
                anomaly_detected_this_frame = True
                
                # --- 新的冷卻處理邏輯 ---
                # 如果當前處於恢復冷卻期
                if memory.anomaly_cooldown_frames > 0:
                    # 強制進入 Coasting 狀態
                    force_coasting_in_cooldown = True
                    # 打印信息
                    print(f"  [冷卻異常] C_Jump: {center_jump:.0f}, W_Ratio: {width_jump_ratio:.2f}. 強制使用記憶。")
                    # 將最終使用的線對設為 None，以觸發後面的 Coasting 邏輯
                    final_pair_to_use = None 
                # 如果冷卻已結束 且 之前不是異常狀態
                elif not memory.is_in_anomaly_state: 
                    # 打印信息
                     print(f"  [異常] C_Jump: {center_jump:.0f}(>{memory.MAX_CENTER_JUMP}), W_Ratio: {width_jump_ratio:.2f}(>{memory.MAX_WIDTH_JUMP_RATIO:.2f}). 進入異常狀態。")
                    # 進入異常狀態
                     memory.is_in_anomaly_state = True 

    # --- 更新記憶與平滑 ---
    
    # 判斷最終是否需要 Coasting (偵測失敗 或 冷卻期強制 或 異常且無可用線對)
    should_coast_now = needs_coasting_due_to_detection_failure or force_coasting_in_cooldown or (memory.is_in_anomaly_state and final_pair_to_use is None)

    # 如果不需要 Coasting 且 找到了最終使用的線對
    if not should_coast_now and final_pair_to_use is not None:
        # 獲取最終線對的參數
        l_params, r_params = final_pair_to_use
        # 只有在非強制 Coasting 時才更新 is_in_anomaly_state
        if not force_coasting_in_cooldown: 
            memory.is_in_anomaly_state = anomaly_detected_this_frame 
        
        # --- 校準階段 ---
        if memory.calibration_frames > 0:
            # 計算寬度和中心點
            current_width = get_bottom_x(image, r_params) - get_bottom_x(image, l_params)
            current_center = (get_bottom_x(image, l_params) + get_bottom_x(image, r_params)) / 2
            # 確保計算成功
            if current_width is not None and current_center is not None: 
                # 初始化或更新基準記憶 (滾動平均)
                if memory.avg_width_bottom is None: 
                    memory.avg_width_bottom = current_width
                    memory.avg_center_bottom = current_center
                else:
                    memory.avg_width_bottom = (memory.avg_width_bottom * 0.9) + (current_width * 0.1)
                    memory.avg_center_bottom = (memory.avg_center_bottom * 0.9) + (current_center * 0.1)
                
                # 校準幀數減一
                memory.calibration_frames -= 1
                # 校準完成時打印信息
                if memory.calibration_frames == 0:
                    print(f"[校準完成] W: {memory.avg_width_bottom:.0f} | C: {memory.avg_center_bottom:.0f}")
                
                # 直接使用當前線對參數更新記憶
                memory.left_params = np.array(l_params)
                memory.right_params = np.array(r_params)
                # 更新 last_good 記憶
                memory.last_good_left_params = memory.left_params 
                memory.last_good_right_params = memory.right_params

        # --- 追蹤階段 ---
        else:
            # 只在 非異常 且 非冷卻 狀態下更新基準記憶
            if not memory.is_in_anomaly_state and memory.anomaly_cooldown_frames <= 0: 
                # 計算當前寬度和中心點
                current_width = get_bottom_x(image, r_params) - get_bottom_x(image, l_params)
                current_center = (get_bottom_x(image, l_params) + get_bottom_x(image, r_params)) / 2
                # 確保計算成功
                if current_width is not None and current_center is not None:
                    # 使用較慢的滾動平均更新基準記憶
                    memory.avg_width_bottom = (memory.avg_width_bottom * 0.95) + (current_width * 0.05) 
                    memory.avg_center_bottom = (memory.avg_center_bottom * 0.95) + (current_center * 0.05) 
                
                # 更新 last_good 記憶 (只有在完全穩定時才更新)
                memory.last_good_left_params = memory.left_params 
                memory.last_good_right_params = memory.right_params

            # 始終更新當前線條參數 (用於平滑)
            if memory.left_params is None: # 如果是第一幀追蹤
                memory.left_params = np.array(l_params)
                memory.right_params = np.array(r_params)
            else: # 使用平滑因子更新
                memory.left_params = (1 - memory.smoothing_factor) * memory.left_params + \
                                     memory.smoothing_factor * np.array(l_params)
                memory.right_params = (1 - memory.smoothing_factor) * memory.right_params + \
                                      memory.smoothing_factor * np.array(r_params)
                
    # 如果需要 Coasting
    elif should_coast_now: 
        # 如果不是因為冷卻期異常 且 之前不是異常狀態，則標記為異常
        if not memory.is_in_anomaly_state and not force_coasting_in_cooldown: 
             memory.is_in_anomaly_state = True 
             print("  [Coasting] 因濾波後無有效線段，使用記憶。")
             
        # 如果存在上一幀的良好記憶
        if memory.last_good_left_params is not None:
            # 使用上一幀的良好記憶來覆蓋當前參數
            memory.left_params = memory.last_good_left_params
            memory.right_params = memory.last_good_right_params
        # (如果連 last_good 都沒有，left/right_params 會保持 None)

    # 返回最終座標 (可能是更新後的，也可能是保持的)
    if memory.left_params is not None and memory.right_params is not None:
        left_line = make_coordinates(image, memory.left_params)
        right_line = make_coordinates(image, memory.right_params)
        return left_line, right_line
    
    # 如果最終無法確定線條，返回 None
    return None, None 

# Canny 邊緣偵測函數
def canny(image):
    # 轉換為灰度圖
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊以去噪
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny 邊緣偵測 (低閾值50, 高閾值150)
    canny_img = cv2.Canny(blur, 50, 150)
    return canny_img

# 定義感興趣區域 (Region of Interest, ROI) 函數
def region_of_interest(image):
    # 獲取影像高度和寬度
    height = image.shape[0]
    width = image.shape[1]
    # 定義梯形區域的頂點座標 (用於遮罩)
    polygons = np.array([
        [(int(width*0.1), height), (int(width*0.45), int(height*0.6)),
         (int(width*0.55), int(height*0.6)), (int(width*0.95), height)]])
    # 建立一個與輸入影像大小相同的黑色遮罩
    mask = np.zeros_like(image)
    # 在遮罩上填充白色梯形區域
    cv2.fillPoly(mask, polygons, 255)
    # 使用遮罩與原始影像進行位元運算，只保留 ROI 區域
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# 繪製綠色車道區域函數
def draw_lane_area(image, left_line, right_line):
    # 建立一個與原始影像大小相同的黑色覆蓋層
    overlay = np.zeros_like(image)
    # 只有在左右線都有效時才繪製
    if left_line is not None and right_line is not None:
        # 獲取左右線的端點座標
        l_x1, l_y1, l_x2, l_y2 = left_line
        r_x1, r_y1, r_x2, r_y2 = right_line
        # 定義四邊形的頂點 (左下, 左上, 右上, 右下)
        pts = np.array([[l_x1, l_y1], [l_x2, l_y2], [r_x2, r_y2], [r_x1, r_y1]], dtype=np.int32)
        # 在覆蓋層上填充綠色四邊形
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
    # 將覆蓋層 (30% 透明度) 與原始影像疊加
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    return result

# --- 主程式 ---
# 設定輸入影片檔名
video_path = 'LaneVedio.mp4'
# 設定輸出影片檔名
output_filename = 'output.mp4' 

# 開啟影片檔案
cap = cv2.VideoCapture(video_path)
# 檢查影片是否成功開啟
if not cap.isOpened(): exit() # 若失敗則退出程式

# 獲取影片的寬度、高度、FPS 和總影格數
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# 處理 FPS 讀取異常的情況 (例如等於0或過大)，設為預設值 30
if fps == 0 or fps > 100: 
    print(f"警告: FPS 讀取異常 ({fps}), 使用預設值 30")
    fps = 30 
    
# 設定影片編碼器為 MP4V (適用於 .mp4)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 建立 VideoWriter 物件，用於寫入輸出影片
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# 打印處理信息
print(f"開始處理影片: {video_path}")
print(f"將儲存至: {output_filename} (加入「冷卻時強制記憶」邏輯)")
print(f"總影格數: {total_frames}, 解析度: {frame_width}x{frame_height}, FPS: {fps:.2f}")

# 初始化影格計數器
frame_count = 0
# 迴圈讀取影片的每一幀
while(cap.isOpened()):
    # 讀取下一幀
    ret, frame = cap.read()
    # 如果讀取失敗 (影片結束或出錯)，則跳出迴圈
    if not ret: break
    # 影格計數器加一
    frame_count += 1
    
    # --- 冷卻計時器 ---
    # 如果恢復冷卻計時器大於 0，則將其減一
    if memory.anomaly_cooldown_frames > 0:
        memory.anomaly_cooldown_frames -= 1
        
    # --- 影像處理流程 ---
    # 1. Canny 邊緣偵測
    canny_image = canny(frame)
    # 2. 套用 ROI 遮罩
    masked_canny = region_of_interest(canny_image)
    # 3. Hough 直線偵測
    lines = cv2.HoughLinesP(masked_canny, rho=2, theta=np.pi/180, threshold=100, 
                            lines=np.array([]), minLineLength=40, maxLineGap=20) 
                            
    # 4. 呼叫主要的車道線追蹤函數，獲取左右線座標
    left_line, right_line = find_and_track_lane_pair(frame, lines, fps) 
    
    # 5. 繪製綠色車道區域
    final_image = draw_lane_area(frame, left_line, right_line)
    # 6. 將處理後的影像寫入輸出影片檔案
    out.write(final_image)

    # 每 30 幀打印一次處理進度
    if frame_count % 30 == 0: print(f"  ... 已處理 {frame_count} / {total_frames} 幀")

# --- 清理 ---
# 釋放影片讀取器
cap.release()
# 釋放影片寫入器
out.release()
# 關閉所有 OpenCV 視窗 (雖然這裡沒顯示)
cv2.destroyAllWindows()
# 打印完成信息
print(f"處理完成！ 影片已儲存至 {output_filename}")
