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
OUTPUT_JSON = 'player_mapping.json'

# === Load YOLOv11 Model ===
yolo_model = YOLO(YOLO_MODEL_PATH)

# === Load ResNet50 for Embeddings ===
resnet = models.resnet50(pretrained=True)
resnet.eval()
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # remove final FC layer

# === Image Preprocessing Transform ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# === Get Embedding Vector from Image ===
def get_embeddings(cropped_img):
    with torch.no_grad():
        img_tensor = transform(cropped_img).unsqueeze(0)
        features = resnet(img_tensor).squeeze().numpy()
        return features / np.linalg.norm(features)  # Normalize

# === Detect Players & Extract Embeddings from a Video ===
def detect_and_embed(video_path):
    cap = cv2.VideoCapture(video_path)
    embeddings = []
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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
    return embeddings, frames

# === Run Detection & Embedding on Both Videos ===
print("Processing broadcast video...")
broadcast_embs, broadcast_frames = detect_and_embed(BROADCAST_VIDEO)

print("Processing tacticam video...")
tacticam_embs, tacticam_frames = detect_and_embed(TACTICAM_VIDEO)

# === Match Players via Cosine Similarity ===
matches = []

print("Matching players across views...")
for i, (tact_emb, tact_box) in enumerate(tacticam_embs):
    max_sim = 0
    matched_id = -1
    for j, (broad_emb, _) in enumerate(broadcast_embs):
        sim = cosine_similarity([tact_emb], [broad_emb])[0][0]
        if sim > max_sim:
            max_sim = sim
            matched_id = j  # You can use UUIDs if needed
    
    matches.append((i, matched_id, max_sim))

# === Save Output to JSON ===
output = []
for i, (tact_idx, matched_broad_idx, score) in enumerate(matches):
    x1, y1, x2, y2 = tacticam_embs[tact_idx][1]
    output.append({
        "player_id": matched_broad_idx,
        "frame": tacticam_frames[tact_idx],
        "bbox": [x1, y1, x2, y2],
        "similarity": float(score)
    })

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Player mapping complete. Output saved to: {OUTPUT_JSON}")
