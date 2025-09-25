import cv2
import numpy as np
from tensorflow import keras

# --- 1. 加载预训练模型 --- [cite: 37]
# 请将 'my_custom_cnn.keras' 替换为您在任务1中保存的模型文件名。
# 这个模型应该是为28x28的单通道灰度图训练的。
try:
    model = keras.models.load_model('my_custom_cnn.keras')
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败，请检查路径和文件是否正确: {e}")
    exit()

# --- 2. 加载视频文件 --- [cite: 36]
# 将 'digits_video.mp4' 替换为您的视频文件名。
# 如果想使用摄像头，请将文件名替换为 0，即 cv2.VideoCapture(0)
video_path = 'digits_video.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"错误：无法打开视频文件 {video_path}")
    exit()

print("视频加载成功，开始处理...")

# --- 3. 循环处理视频的每一帧 --- [cite: 38]
while True:
    # 读取下一帧 [cite: 38]
    ret, frame = cap.read()
    
    # 如果视频结束或读取失败，则退出循环
    if not ret:
        print("视频处理完成。")
        break

    # --- 4. 图像预处理 --- 
    # 将帧转换为灰度图
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) [cite: 47]
    
    # 使用高斯模糊减少噪声，有助于轮廓检测
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # 使用阈值处理将图像二值化 [cite: 49]
    # 我们假设数字是白色（或亮色），背景是黑色（或暗色）。
    # THRESH_BINARY_INV 会将亮区域变为黑色，暗区域变为白色，方便模型处理。
    # 127 是阈值，可以根据视频的光照条件进行调整。
    _, thresh_frame = cv2.threshold(blurred_frame, 127, 255, cv2.THRESH_BINARY_INV)

    # --- 5. 检测数字轮廓 --- 
    # 查找二值化图像中的所有轮廓
    contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有找到的轮廓
    for contour in contours:
        # 获取每个轮廓的边界框
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # 过滤掉太小或太大的轮廓，避免噪声干扰
        if w >= 15 and h >= 15:
            
            # --- 6. 提取、处理并输入ROI到模型 --- [cite: 40]
            # 从阈值图像中提取数字的ROI (Region of Interest)
            roi = thresh_frame[y:y+h, x:x+w]
            
            # 将ROI大小调整为模型所需的28x28 [cite: 46]
            roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            
            # 归一化并调整维度以匹配模型输入 (1, 28, 28, 1)
            roi_normalized = roi_resized.astype('float32') / 255.0
            roi_input = np.reshape(roi_normalized, (1, 28, 28, 1))
            
            # 使用模型进行预测
            prediction = model.predict(roi_input)
            predicted_value = np.argmax(prediction)
            probability = np.max(prediction)
            
            # --- 7. 在原始帧上绘制结果 --- 
            # 在数字周围绘制矩形框
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) [cite: 51]
            
            # 准备要显示的标签文本
            label = f"Digit: {predicted_value}"
            prob_label = f"Prob: {probability:.2f}"
            
            # 将预测值和概率显示在框的上方
            cv2.putText(frame, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) [cite: 50]
            cv2.putText(frame, prob_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) [cite: 50]

    # --- 8. 显示处理后的帧 --- 
    cv2.imshow('Handwritten Digit Recognition', frame) [cite: 52]

    # 等待按键，如果按下 'q' 键则退出循环 [cite: 53]
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 9. 释放资源 --- [cite: 43]
cap.release()
cv2.destroyAllWindows() [cite: 54]