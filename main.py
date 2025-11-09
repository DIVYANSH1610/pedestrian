import cv2
import torch
import os
import time
import numpy as np   
from ultralytics import YOLO
from shapely.geometry import Polygon, Point

video_source = "data/sample_video.mp4"
save_path = "data/snapshots"
model_path = "yolov8n.pt"

os.makedirs(save_path, exist_ok=True)
model = YOLO(model_path)
print("✅ YOLOv8 model loaded successfully!")

zone_points = [(100, 400), (500, 400), (500, 600), (100, 600)]  
zone_points = [(100, 450), (600, 450), (700, 300), (200, 300)]
zone_polygon = Polygon(zone_points)

cap = cv2.VideoCapture(video_source)
if not cap.isOpened():
    print("❌ Cannot open video source.")
    exit()

print("🚦 Detection started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)

    cv2.polylines(frame, [np.array(zone_points, np.int32)], True, (0, 255, 255), 2)
    cv2.putText(frame, "Pedestrian Zone", (zone_points[0][0], zone_points[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]

            if label in ["car", "bus", "truck", "motorcycles","pedestrian"]:
                x1, y1, x2, y2 = box.xyxy[0]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                point = Point(cx, cy)
                if zone_polygon.contains(point):
                    color = (0, 0, 255) 
                    cv2.putText(frame, "🚫 Vehicle in Pedestrian Zone!", (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(save_path, f"violation_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                else:
                    color = (0, 255, 0) 

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Pedestrian Zone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Detection stopped.")
