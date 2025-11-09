# 🚦 Pedestrian Zone Vehicle Entry Detection

## 📌 Problem Statement
Urban pedestrian-only zones often face unauthorized vehicle entries, causing safety risks and congestion.  
This project develops an AI-based system to **detect vehicles entering restricted pedestrian zones** using **real-time computer vision** and **YOLOv8 deep learning**.

---

## 🎯 Objective
- Detect vehicles entering pedestrian-restricted zones in real-time.
- Use **YOLOv8** for object detection.
- Capture snapshots and log violations.
- (Optional) Use OCR for license plate extraction.

---

## ⚙️ Parameters Considered
| Parameter | Description |
|------------|-------------|
| Vehicle Detection Accuracy | Correctly identifying vehicles using YOLOv8 |
| Zone Violation Detection | Detect when a vehicle crosses the restricted zone |
| FPS (Speed) | Real-time detection capability |
| Lighting | Works under different brightness |
| Camera Angle | Handles different camera positions |

---

## 🧠 Technologies Used
- **Python 3.10+**
- **OpenCV**
- **Ultralytics YOLOv8**
- **PyTorch**
- **Shapely**
- **EasyOCR**
- **NumPy, Pillow, Imutils**

---

## 🧩 Solution Steps
1. **Collect Data:** Videos/images of pedestrian zones  
2. **Preprocess:** Resize frames, define restricted zone ROI  
3. **Model Selection:** YOLOv8 (pre-trained on COCO)  
4. **Detection:** Track vehicles frame by frame  
5. **Violation Capture:** If ROI breached → save frame  
6. **Log Results:** Store images + timestamps

---

## 🧾 Dataset
Dataset link: [Google Drive Folder](https://drive.google.com/drive/folders/1cMa_GfbsatWe-flCB4TZ9uiGEfW1PKhR?usp=drive_link)

Structure:
