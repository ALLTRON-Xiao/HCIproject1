import cv2
import numpy as np
from tensorflow import keras

# load model

try:
    model = keras.models.load_model('cnn_dropout_adam2.keras')
    print("success")
except Exception as e:
    print(f"Error: check model path: {e}")
    exit()


video_path = 'small-n.mp4' 
cap = cv2.VideoCapture(video_path)
# # if use camera use this
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(f"Error can't open video: {video_path}")
    exit()

print("load video success")

# loop to handle each frame
while True:

    ret, frame = cap.read()    
    if not ret:
        print("video over")
        break


    # to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # threshdle processing and reverse color
    # set 80 for my camera, 185 better for example video
    _, thresh_frame = cv2.threshold(blurred_frame, 185, 255, cv2.THRESH_BINARY_INV)


    # Closing(dilation,erosion)
    kernel = np.ones((7,7), np.uint8)
    closing_frame = cv2.morphologyEx(thresh_frame, cv2.MORPH_CLOSE, kernel)
    
    closing_frame = cv2.resize(closing_frame, (720, 540), interpolation = cv2.INTER_AREA)
    cv2.imshow('Grayscale Frame', closing_frame)

    # identify outline
    contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        
        if w >= 15 and h >= 15:
            
            roi = thresh_frame[y:y+h, x:x+w]
            
            roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            
            roi_normalized = roi_resized.astype('float32') / 255.0
            roi_input = np.reshape(roi_normalized, (1, 28, 28, 1))
            
            prediction = model.predict(roi_input)
            predicted_value = np.argmax(prediction)
            probability = np.max(prediction)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            label = f"{predicted_value}"
            prob_label = f"{probability:.2f}"
            
            cv2.putText(frame, label, (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, prob_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    frame = cv2.resize(frame, (1080, 720), interpolation = cv2.INTER_AREA)
    cv2.imshow('Handwritten Digit Recognition', frame)

    # key q for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # # for each frame stop debug
    # # 
    # key = cv2.waitKey(0) & 0xFF

    # # if q stop
    # if key == ord('q'):
    #     break
    # # if anyother key, next frame
    # else:
    #     continue

cap.release()
cv2.destroyAllWindows()