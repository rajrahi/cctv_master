import cv2
import numpy as np
import time

class VideoConfig:
    def __init__(self, source=0, target_fps=30):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.previous_frame = None
        
        self.set_fps(target_fps)  # Set target fps internally
        self.last_frame_time = 0  # Time when last frame was read
    
    def set_resolution(self, width, height):
        """Set the resolution of the video capture"""
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def set_fps(self, fps):
        """Set the FPS of the video capture and store target fps"""
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.target_fps = fps
        self.frame_time = 1.0 / fps if fps > 0 else 0
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_current_settings(self):
        """Get current video settings"""
        return {
            'width': self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            'height': self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'target_fps': self.target_fps
        }

    def detect_motion(self, frame, threshold=30):
        """Detect motion in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return False
        
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        thresh = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.previous_frame = gray
        
        return len(contours) > 0

    def show_frame(self, frame, window_name='Frame', show=True):
        """Display the frame if show is True"""
        if show:
            cv2.imshow(window_name, frame)
            # waitKey will be controlled outside for frame timing

    def read_frame(self):
        """Read a frame at the target fps rate"""
        current_time = time.time()
        time_since_last = current_time - self.last_frame_time
        
        # If we are too fast, sleep to maintain FPS
        if time_since_last < self.frame_time:
            time.sleep(self.frame_time - time_since_last)
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.last_frame_time = time.time()
        return frame

    def release(self):
        """Release the video capture"""
        if self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()





# import cv2
# from time import sleep

# # Import the class (assume it's in video_config.py or defined above)
# # from video_config import VideoConfig

# def main():
#     # Initialize video config with default webcam (source=0)
#     vc = VideoConfig(source=0)
    
#     # Set resolution and FPS
#     vc.set_resolution(640, 480)
#     vc.set_fps(30)

#     print("Video settings:", vc.get_current_settings())

#     try:
#         while True:
#             ret, frame = vc.cap.read()
#             if not ret:
#                 print("Failed to grab frame")
#                 break
            
#             # Detect motion
#             if vc.detect_motion(frame):
#                 print("Motion Detected")

#             # Show the current frame
#             vc.show_frame(frame)

#     except KeyboardInterrupt:
#         print("Stopped by user")

#     finally:
#         vc.release()
#         print("Video capture released.")

# if __name__ == "__main__":
#     main()
