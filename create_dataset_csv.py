import os
import csv

# Path to your data folder
data_dir = 'data'
output_csv = os.path.join(data_dir, 'dataset_info.csv')

# ✅ Automatically detect all folders (vehicle types) in 'data'
categories = [
    f for f in os.listdir(data_dir)
    if os.path.isdir(os.path.join(data_dir, f)) and f.lower() != "snapshots"
]

# ✅ Create dataset_info.csv
with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Class', 'Image_Name', 'Path'])

    for category in categories:
        class_folder = os.path.join(data_dir, category)
        for img_file in os.listdir(class_folder):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                writer.writerow([category, img_file, os.path.join(class_folder, img_file)])

print(f"✅ Dataset CSV created successfully at: {output_csv}")
