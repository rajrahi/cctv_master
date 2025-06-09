import threading
from time import sleep
from number_person_detection.number_of_person import PersonDetector


import traceback
class model():
    def __init__(self):
        # self.name = name
        self.detections = []
        self.resolution = (1080, 720)
        self.fps = 30

    def add_number_of_person_detection(self , reqs=None ,id=None):
        
        
        
        person_detector = PersonDetector(source="rtsp://admin:tech@9900@106.51.129.154:554/Streaming/Channels/201" , id=id)
        thread = threading.Thread(target=person_detector.detect)
        thread.start()


        self.detections.append(person_detector)         


    def add_motion_detection(self, reqs=None):
        # Placeholder for motion detection logic
        pass