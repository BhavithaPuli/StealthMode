import cv2
import torch
import numpy as np
import json
from ultralytics import YOLO
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

# === Config ===
YOLO_MODEL_PATH = 'best.pt'
BROADCAST_VIDEO = 'broadcast.mp4'
TACTICAM_VIDEO = 'tacticam.mp4'
OUTPUT_JSON = "C:/temp/playermapping.json"
SIMILARITY_WEIGHT = 0.8
# === Device Selection ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n Using device: {device}")

# === Load YOLOv11 Model ===
yolo_model = YOLO(YOLO_MODEL_PATH)  # YOLO handles device internally

# === Load ResNet50 for Embeddings ===
resnet = models.resnet50(pretrained=True)
resnet.eval()
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # remove final FC layer
resnet = resnet.to(device)

# === Image Preprocessing Transform ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Get Embedding Vector from Image ===
def get_embeddings(cropped_img):
    with torch.no_grad():
        img_tensor = transform(cropped_img).unsqueeze(0).to(device)
        features = resnet(img_tensor).squeeze().cpu().numpy()
        return features / np.linalg.norm(features)

# === Detect Players & Extract Embeddings from a Video ===
def detect_and_embed(video_path, label):
    cap = cv2.VideoCapture(video_path)
    embeddings = []
    frames = []
    frame_idx = 0
    print(f"\n Starting detection on: {label}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        print(f"  Processing frame {frame_idx} from {label}...")
        results = yolo_model(frame, verbose=False)[0]
        for box in results.boxes.data:
            x1, y1, x2, y2, conf, cls = map(int, box[:6])
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            emb = get_embeddings(cropped)
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            embeddings.append((emb, (x1, y1, x2, y2)))
            frames.append(frame_number)
    cap.release()
    print(f" Finished {label}: {frame_idx} frames, {len(embeddings)} players detected.\n")
    return embeddings, frames

# === Run Detection & Embedding ===
broadcast_embs, broadcast_frames = detect_and_embed(BROADCAST_VIDEO, "broadcast.mp4")
tacticam_embs, tacticam_frames = detect_and_embed(TACTICAM_VIDEO, "tacticam.mp4")

# === Prepare matrices for vectorized matching ===
tact_feats = np.stack([e for e, _ in tacticam_embs])
broad_feats = np.stack([e for e, _ in broadcast_embs])
visual_sims = tact_feats @ broad_feats.T  # cosine similarity matrix (N, M)

tact_centers = np.array([((x1+x2)/2, (y1+y2)/2) for _, (x1,y1,x2,y2) in tacticam_embs])
broad_centers = np.array([((x1+x2)/2, (y1+y2)/2) for _, (x1,y1,x2,y2) in broadcast_embs])
diffs = tact_centers[:, None, :] - broad_centers[None, :, :]
dists = np.linalg.norm(diffs, axis=2)
max_dim = max([x2-x1 for _,(x1,y1,x2,y2) in tacticam_embs + broadcast_embs] + [1])
spatial_sims = 1 - np.clip(dists / max_dim, 0, 1)

final_sims = SIMILARITY_WEIGHT * visual_sims + (1 - SIMILARITY_WEIGHT) * spatial_sims

# === Match Players ===
matches = np.argmax(final_sims, axis=1)
scores = np.max(final_sims, axis=1)

print(f"\n Matching complete: tacticam {len(matches)} ↔ broadcast {len(broadcast_embs)}\n")

# === Save to JSON ===
output = []
for i, match_idx in enumerate(matches):
    x1, y1, x2, y2 = tacticam_embs[i][1]
    output.append({
        "player_id": int(match_idx),
        "frame": tacticam_frames[i],
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "similarity": float(scores[i])
    })
try:
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Player mapping saved to '{OUTPUT_JSON}'\n")
except Exception as e:
    print(f"Error saving JSON: {e}")

