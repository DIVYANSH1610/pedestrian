from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # This will auto-download weights
print("✅ YOLO model loaded successfully!")
