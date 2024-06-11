import cv2
import numpy as np
import time
import requests
from playsound import playsound
import threading

# Constants for time calculation
TIME_PER_PEDESTRIAN = 2  
TIME_PER_VEHICLE = 1.5  
MIN_RED_TIME = 10  
MAX_RED_TIME = 45  
ALERT_SOUND = 'audio.mp3'  
# URL to Flask app data endpoint
FLASK_DATA_URL = 'http://127.0.0.1:5000/data'

# Load pre-trained models
vehicle_model = cv2.dnn.readNet('./yolov3.weights', './yolov3.cfg')

# Get output layer names
layer_names = vehicle_model.getLayerNames()
output_layers = [layer_names[i - 1] for i in vehicle_model.getUnconnectedOutLayers()]

# Load class names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture (0 for webcam or provide a video file path)
cap = cv2.VideoCapture('videoo.mp4')

def play_alert_sound():
    playsound(ALERT_SOUND)

def calculate_red_signal_duration(pedestrian_count, vehicle_count, 
                                  base_duration=20, adjustment_pedestrian=0.5, adjustment_vehicle=0.3, 
                                  min_duration=5, max_duration=90):
    # Calculate adjustments
    T_P = pedestrian_count * adjustment_pedestrian
    T_V = vehicle_count * adjustment_vehicle
    
    # Compute total red signal duration
    T_red = base_duration + T_P + T_V
    
    # Ensure duration is within limits
    T_red = max(min_duration, min(T_red, max_duration))
    
    return T_red

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # Skip frames to reduce processing load
    if frame_count % 5 != 0:
        continue

    # Prepare the frame for vehicle detection
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    vehicle_model.setInput(blob)
    detections = vehicle_model.forward(output_layers)
    
    # Initialize counts
    pedestrian_count = 0
    vehicle_count = 0
    
    # To keep track of already counted pedestrians (centroids)
    detected_pedestrians = []
    
    # Lists to hold detection data for NMS
    boxes = []
    confidences = []
    class_ids = []
    
    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Get coordinates for the bounding box
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                # Save detection data
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x, y, w, h = box
            class_id = class_ids[i]
            label = str(classes[class_id])
            
            # Check if the detected object is a pedestrian
            if label == 'person':
                # Check for double counting
                center_x = x + w // 2
                center_y = y + h // 2
                if not any((abs(center_x - px) < w / 2 and abs(center_y - py) < h / 2) for px, py in detected_pedestrians):
                    pedestrian_count += 1
                    detected_pedestrians.append((center_x, center_y))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Check if the detected object is a vehicle
            elif label in ['car', 'truck', 'bus', 'motorbike']:
                vehicle_count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Calculate red light duration based on pedestrian and vehicle count
    total_red_time = calculate_red_signal_duration(pedestrian_count, vehicle_count)
    
    # Update shared data for Flask app asynchronously
    data = {
        "pedestrian_count": pedestrian_count,
        "vehicle_count": vehicle_count,
        "red_light_duration": total_red_time
    }
    threading.Thread(target=requests.post, args=(FLASK_DATA_URL,), kwargs={'json': data}).start()
    
    # Display counts and red light time on frame
    cv2.putText(frame, f'Pedestrians: {pedestrian_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f'Vehicles: {vehicle_count}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Red Light Time: {total_red_time:.1f} sec', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display red light visual feedback
    cv2.circle(frame, (width - 50, 50), 20, (0, 0, 255), -1)
    
    # Play red light alert if the red light duration is above a threshold
    if total_red_time > 5:  # Adjust the threshold as needed
        threading.Thread(target=play_alert_sound).start()
    
    # Show the frame
    cv2.imshow('Frame', frame)
    
    # Check for key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
