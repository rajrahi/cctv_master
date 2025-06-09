import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

# Load all known face embeddings from a directory
def load_known_faces(embedding_dir, device):
    known_faces = {}
    
    for file in os.listdir(embedding_dir):

        if file.endswith(".pt"):
            
            name = os.path.splitext(file)[0]
            embedding_path = os.path.join(embedding_dir, file)
            try:
                embedding = torch.load(embedding_path).to(device)  # Move to device
                known_faces[name] = embedding
            except Exception as e:
                print(f"Failed to load {file}: {e}")


    print(f"Loaded {len(known_faces)} known faces.")
    return known_faces

# Convert face image to 512-dim embedding
def get_face_embedding(face_img, model, device):
    face_img = face_img.resize((160, 160))
    face_tensor = torch.tensor(np.array(face_img)).permute(2, 0, 1).float().to(device) / 255.0
    face_tensor = face_tensor.unsqueeze(0)
    with torch.no_grad():
        embedding = model(face_tensor)
    return embedding.squeeze()

import torch.nn.functional as F

# Compare input embedding to known faces using cosine similarity
def recognize_face(embedding, known_faces, threshold=0.5):
    best_match = "Unknown"
    best_similarity = -1.0  # Cosine similarity ranges from -1 to 1

    for name, known_embedding in known_faces.items():
        similarity = F.cosine_similarity(embedding.unsqueeze(0), known_embedding.unsqueeze(0)).item()
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = name

    if best_similarity < threshold:
        return "Unknown", best_similarity
    return best_match, best_similarity

def main():
    show_window = True  # Toggle this flag

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    detector = MTCNN(keep_all=True, device=device)
    embedder = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    known_faces = load_known_faces("embeddings", device)

    cap = cv2.VideoCapture("./input.mp4")
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = detector.detect(rgb_frame)

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                x1, y1, x2, y2 = map(int, box)
                face = rgb_frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_pil = Image.fromarray(face)
                embedding = get_face_embedding(face_pil, embedder, device)
                name, distance = recognize_face(embedding, known_faces)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({distance:.2f})", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if show_window:
            cv2.imshow("Face Authentication", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
