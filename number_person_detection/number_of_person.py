import sys
import os
from time import sleep
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from opencv import VideoConfig
import cv2
import numpy as np
from openvino.runtime import Core

import threading

class PersonDetector:
    def __init__(self, source, target_fps=30, width=640, height=480, ishow=True):
        self.ie = Core()
        self.model = self.ie.read_model("number_person_detection/models/person-detection-retail-0013.xml")
        self.compiled_model = self.ie.compile_model(self.model, "CPU")
        self.input_layer = self.compiled_model.input(0)

        self.vc = VideoConfig(source=source, target_fps=target_fps)
        self.vc.set_resolution(width, height)
        self.width = width
        self.height = height
        self.ishow = ishow

        self.paused = False
        self._stop = False
        self._lock = threading.Lock()

    def pause(self):
        with self._lock:
            if not self.paused:
                self.paused = True
                print("[INFO] Paused due to no motion")

    def resume(self):
        with self._lock:
            if self.paused:
                self.paused = False
                print("[INFO] Resumed due to motion detected")

    def stop(self):
        with self._lock:
            self._stop = True
            print("[INFO] Stopping detection")

    def detect(self):
        paused_frame = None
        import psutil
        import GPUtil
        import logging

        # Configure logging
        logging.basicConfig(filename='system_metrics.log', level=logging.INFO, 
                          format='%(asctime)s - %(message)s')

        while True:
            # Log system metrics
            cpu_percent = psutil.cpu_percent()
            ram = psutil.virtual_memory()
            try:
                gpus = GPUtil.getGPUs()
                gpu_load = gpus[0].load * 100 if gpus else 0
                gpu_memory = gpus[0].memoryUsed if gpus else 0
            except:
                gpu_load = 0
                gpu_memory = 0
            
            logging.info(f"CPU Usage: {cpu_percent}% | RAM Usage: {ram.percent}% | GPU Load: {gpu_load}% | GPU Memory: {gpu_memory}MB")

            with self._lock:
                if self._stop:
                    break
                local_paused = self.paused

            frame = self.vc.read_frame()
            if frame is None:
                break

            if not self.vc.detect_motion(frame):
                self.pause()
            else:
                self.resume()

            if not local_paused:
                frame = cv2.resize(frame, (self.width, self.height))
                resized = cv2.resize(frame, (544, 320))
                input_image = np.expand_dims(resized.transpose(2, 0, 1), 0)

                output = self.compiled_model([input_image])[self.compiled_model.output(0)]

                person_count = 0
                for det in output[0][0]:
                    _, label, conf, x_min, y_min, x_max, y_max = det
                    if conf > 0.5:
                        person_count += 1
                        cv2.rectangle(frame, 
                                    (int(x_min * frame.shape[1]), int(y_min * frame.shape[0])),
                                    (int(x_max * frame.shape[1]), int(y_max * frame.shape[0])),
                                    (0, 255, 0), 2)

                print(f"Number of persons detected: {person_count}")
                paused_frame = frame.copy()
            else:
                frame = paused_frame

            if self.ishow and frame is not None:
                cv2.imshow("Person Detection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break        
        self.vc.cap.release()
        cv2.destroyAllWindows()


# detector = PersonDetector(source="rtsp://admin:tech@9900@106.51.129.154:554/Streaming/Channels/201")
# thread = threading.Thread(target=detector.detect)
# thread.start()

# sleep(5)  # Allow some time for detection to start

# # Later from another thread or main process
# detector.pause()    # pauses detection
# # detector.resume()   # resumes detection
# # detector.stop()     # stops detection loop






# import cv2
# from ultralytics import YOLO

# def detect_persons_yolo():
#     # Load YOLOv8 model (you can use 'yolov8n.pt' for faster, or 'yolov8m.pt' for more accuracy)
#     model = YOLO('yolov8n.pt')  # or yolov8s.pt/yolov8m.pt/yolov8l.pt

#     # Connect to RTSP stream
#     cap = cv2.VideoCapture("rtsp://admin:tech@9900@106.51.129.154:554/Streaming/Channels/201", cv2.CAP_FFMPEG)

#     while True:
#         ret, frame = cap.read()
#         frame = cv2.resize(frame, (640, 480))
#         if not ret:
#             print("Failed to grab frame")
#             break

      
#         results = model(frame, classes=[0])  # class 0 = person

     
#         annotated_frame = results[0].plot()

    
#         num_persons = sum(1 for c in results[0].boxes.cls if int(c) == 0)

#         print(f"Number of persons detected: {num_persons}" )

#     #     if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         break

#     # cap.release()
#     # cv2.destroyAllWindows()

# if __name__ == "__main__":
#     detect_persons_yolo()


