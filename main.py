import cv2
import torch
import os
import time
import numpy as np 
from ultralytics import YOLO
from shapely.geometry import Polygon, Point
from datetime import datetime

# --- CONFIGURATION ---
video_source = "data/samplevideo.mp4git" # provide the video 
save_path = "data/snapshots"
# NOTE: Use your best trained model path here (e.g., train11/weights/best.pt)
model_path = "yolov8n.pt" 

os.makedirs(save_path, exist_ok=True)
model = YOLO(model_path)
print("✅ YOLOv8 model loaded successfully!")

# --- PEDESTRIAN ZONE DEFINITION ---
# NOTE: The last one overwrites the first. Use the correct, finalized coordinates.
zone_points = [(100, 450), (600, 450), (700, 300), (200, 300)]
zone_polygon = Polygon(zone_points)

cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("❌ Cannot open video source.")
    exit()

print("🚦 Detection started... Press 'q' to quit.")

# --- TRACKING VIOLATION STATE ---
last_violation_time = 0
VIOLATION_COOLDOWN = 2 # Seconds between saving snapshots

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flag to check if ANY violation occurred in the current frame
    violation_detected_in_frame = False 
    
    # 1. RUN DETECTION
    results = model(frame, stream=True, verbose=False) # Add verbose=False for cleaner output

    # Draw the zone
    cv2.polylines(frame, [np.array(zone_points, np.int32)], True, (0, 255, 255), 2)
    cv2.putText(frame, "Pedestrian Zone", (zone_points[0][0], zone_points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            # 2. SEPARATE LOGIC: Check ONLY restricted vehicle types
            if label in ["car", "bus", "truck", "motorcycles", "lorry", "small lorry"]: # ADD ALL RESTRICTED VEHICLES HERE
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                point = Point(cx, cy)
                
                # 3. CORE LOGIC: Check for intersection
                if zone_polygon.contains(point):
                    color = (0, 0, 255) # RED
                    violation_detected_in_frame = True
                else:
                    color = (0, 255, 0) # GREEN

                # Draw bounding box and circle (common to all vehicle types)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw pedestrians in a neutral color (e.g., blue) if needed
            elif label == "pedestrian":
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                color = (255, 0, 0) # BLUE
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4. SNAPSHOT AND ALERT (OUTSIDE INNER LOOP)
    if violation_detected_in_frame:
        # Display large violation alert
        cv2.putText(frame, "🚫 VIOLATION DETECTED! 🚫", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # Check cooldown before saving snapshot
        current_time = time.time()
        if current_time - last_violation_time > VIOLATION_COOLDOWN:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_path, f"violation_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            last_violation_time = current_time # Reset timer
            print(f"📸 Snapshot saved: {filename}")


    cv2.imshow("Pedestrian Zone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Detection stopped.")