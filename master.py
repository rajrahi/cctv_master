from time import sleep
from model import model
import torch
from multiprocessing import Process

# Check for GPU availability
if torch.cuda.is_available():
    print("CUDA is available.")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available.")

# Create model instance
model_instance = model()

# Function to run model logic
def run_instance():
    model_instance.add_number_of_person_detection()

# Main entry
if __name__ == "__main__":
    processes = []

    for i in range(10):
        print(f"Starting instance {i+1}")
        p = Process(target=run_instance)
        p.start()
        processes.append(p)

    for i, p in enumerate(processes):
        p.join()
        print(f"Instance {i+1} finished.")
