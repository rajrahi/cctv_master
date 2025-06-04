import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import psutil
import GPUtil
import time
import os
import torch

def get_system_metrics():
    cpu_percent = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    gpu_info = []
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            gpu_info.append({
                'gpu_load': gpu.load * 100,
                'gpu_memory_used': gpu.memoryUsed,
                'gpu_memory_total': gpu.memoryTotal,
                'gpu_util': gpu.memoryUtil * 100
            })
    except:
        gpu_info = [{'gpu_load': 0, 'gpu_memory_used': 0, 'gpu_memory_total': 0, 'gpu_util': 0}]
    
    return cpu_percent, ram, disk, gpu_info

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load YOLOv8 model
start_time = time.time()
model = YOLO('yolov8s.pt')
model.to(device)  # Move model to GPU
model_load_time = time.time() - start_time
print("\n=== Model Loading Metrics ===")
print(f"Model Loading Time: {model_load_time:.2f} seconds")

# Initialize video capture from RTSP stream
cap = cv2.VideoCapture("rtsp://admin:tech@9900@106.51.129.154:554/Streaming/Channels/201")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print("\n=== Camera Properties ===")
print(f"Resolution: {frame_width}x{frame_height}")
print(f"FPS: {fps}")

# Initialize video writer
out = cv2.VideoWriter('processed_livestream.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps//3, (frame_width, frame_height))

# Performance metrics
frame_count = 0
processing_start_time = time.time()
frames_processed = 0

while True:
    frame_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    frame_count += 1
    if frame_count % 3 != 0:  # Process every 3rd frame
        continue

    # Convert frame to tensor and move to GPU
    frame_tensor = torch.from_numpy(frame).to(device)
    results = model(frame_tensor, imgsz=frame.shape[:2])
    head_detections = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            if class_id == 0 and confidence > 0.1:
                box_height = y2 - y1
                if box_height < 20:
                    continue

                head_y = y1 + int(box_height * 0.15)
                head_x = x1 + (x2 - x1) // 2
                centroid = (head_x, head_y)

                head_detections.append({'centroid': centroid})
                cv2.circle(frame, centroid, 3, (0, 255, 0), -1)

    centroids = np.array([d['centroid'] for d in head_detections])
    groups = []

    if len(centroids) > 0:
        clustering = DBSCAN(eps=80, min_samples=4).fit(centroids)
        labels = clustering.labels_
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                continue
            group_indices = [i for i, lbl in enumerate(labels) if lbl == label]
            if len(group_indices) >= 10:
                groups.append(group_indices)

    merged_groups = []
    used = set()

    for i, group1 in enumerate(groups):
        if i in used:
            continue
        merged = set(group1)
        for j, group2 in enumerate(groups):
            if i != j and j not in used:
                pts1 = np.array([head_detections[k]['centroid'] for k in group1])
                pts2 = np.array([head_detections[k]['centroid'] for k in group2])
                center1 = np.mean(pts1, axis=0)
                center2 = np.mean(pts2, axis=0)
                dist = np.linalg.norm(center1 - center2)
                if dist < 100:
                    merged.update(group2)
                    used.add(j)
        merged_groups.append(list(merged))
        used.add(i)

    y_offset = 30
    cv2.putText(frame, f'Total Crowds: {len(merged_groups)}', (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    for idx, group in enumerate(merged_groups, start=1):
        points = np.array([head_detections[i]['centroid'] for i in group])
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)

        cv2.rectangle(frame, (int(x_min)-10, int(y_min)-10), (int(x_max)+10, int(y_max)+10), (0, 0, 255), 2)
        label = f'Crowd {idx}: {len(group)} persons'
        cv2.putText(frame, label, (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        y_offset += 30
        cv2.putText(frame, label, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 255), 2)

    out.write(frame)
    cv2.imshow('Crowd Detection', frame)

    frames_processed += 1
    if frames_processed % 30 == 0:  # Print metrics every 30 frames
        cpu_percent, ram, disk, gpu_info = get_system_metrics()
        print("\n=== System Resource Usage ===")
        print(f"CPU Usage: {cpu_percent}%")
        print(f"RAM Usage: {ram.percent}% (Used: {ram.used/1024/1024:.1f}MB)")
        print(f"Disk Usage: {disk.percent}% (Used: {disk.used/1024/1024/1024:.1f}GB)")
        for idx, gpu in enumerate(gpu_info):
            print(f"GPU {idx} Load: {gpu['gpu_load']:.1f}%")
            print(f"GPU {idx} Memory: {gpu['gpu_memory_used']:.1f}MB / {gpu['gpu_memory_total']:.1f}MB")
            print(f"GPU {idx} Utilization: {gpu['gpu_util']:.1f}%")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

total_time = time.time() - processing_start_time
fps_achieved = frames_processed / total_time

print("\n=== Final Processing Report ===")
print(f"Total Processing Time: {total_time:.2f} seconds")
print(f"Frames Processed: {frames_processed}")
print(f"Average FPS: {fps_achieved:.2f}")
print(f"Output Video Size: {os.path.getsize('processed_livestream.mp4') / (1024*1024):.2f} MB")

# Clean up
torch.cuda.empty_cache()  # Clear GPU memory
cap.release()
out.release()
cv2.destroyAllWindows()