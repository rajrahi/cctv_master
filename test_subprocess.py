# my_class.py
import time

class MyWorker:
    def __init__(self, name, count):
        self.name = name
        self.count = count

    def run(self):
        for i in range(self.count):
            print(f"{self.name} -> {i}")
            time.sleep(1)



# run_instances.py
from multiprocessing import Process
# from my_class import MyWorker

def run_instance(name, count):
    worker = MyWorker(name, count)
    worker.run()

if __name__ == "__main__":
    p1 = Process(target=run_instance, args=("Worker-1", 5))
    p2 = Process(target=run_instance, args=("Worker-2", 3))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
