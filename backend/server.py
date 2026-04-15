from sam_segment import extract_leaf
from fastapi import FastAPI, File, UploadFile, Form
import shutil
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
import uvicorn

app = FastAPI()

# =========================
# PARAMETERS
# =========================

IMAGE_SIZE = (320, 320)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_CONFIGS = {
    "wheat": {
        "model_path": "models/leaf_disease_rgb_only_resnet18NEW.pth",
        "classes": ["Yellow rust", "Healthy", "Brown rust"]
    },
    "rice": {
        "model_path": "/content/drive/MyDrive/leaf-ai-api/models/leaf_disease_rgb_only_resnet(RICE).pth",
        "classes": ["Blast", "Bacterial blight", "Brown spot"]
    },
    "sugarcane": {
        "model_path": "/content/drive/MyDrive/leaf-ai-api/models/sugarcane_resnet18.pth",
        "classes": ["MOSAIC", "REDROT", "RUST"]
    }
}

# =========================
# LOAD MODELS
# =========================

def load_model(model_path, num_classes):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

loaded_models = {}

for crop_name, config in MODEL_CONFIGS.items():
    print(f"Loading {crop_name} model...", flush=True)
    loaded_models[crop_name] = load_model(
        config["model_path"],
        len(config["classes"])
    )

print("✅ All CNN models loaded")

# =========================
# API ENDPOINT
# =========================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    crop_type: str = Form(...)
):
    crop_type = crop_type.strip().lower()

    if crop_type not in MODEL_CONFIGS:
        return {
            "error": "Invalid crop_type. Use 'wheat' or 'rice'"
        }

    print("1. Request received", flush=True)
    print("Crop type:", crop_type, flush=True)

    save_path = "received_image.jpg"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("2. Image saved", flush=True)

    img = cv2.imread(save_path)
    print("3. Image read", flush=True)

    img = extract_leaf(img)
    print("4. extract_leaf finished", flush=True)

    cv2.imwrite("segmented_leaf.jpg", img)
    print("5. Segmented image saved", flush=True)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMAGE_SIZE)
    print("6. Image resized", flush=True)

    img_rgb = img_rgb.astype(np.float32) / 255.0
    input_tensor = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    print("7. Tensor created", flush=True)

    model = loaded_models[crop_type]
    classes = MODEL_CONFIGS[crop_type]["classes"]

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    print("8. Prediction done", flush=True)

    pred_class = classes[pred_idx.item()]
    confidence = float(conf.item() * 100)

    print("9. Returning response", flush=True)

    return {
        "crop_type": crop_type,
        "disease": pred_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)