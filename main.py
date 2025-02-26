import torch
from transformers import ViTForImageClassification, ViTConfig
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import uvicorn

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = ViTConfig(num_labels=4)  
model = ViTForImageClassification(config).to(device)
model.load_state_dict(torch.load("final_ssl_soilnet_model.pth", map_location=device))
model.eval()

# Define class labels
class_labels = ["Alluvial soil", "Black Soil", "Clay soil", "Red soil"]

# Crop suggestions
crop_suggestions = {
    "Alluvial soil": ["Wheat", "Rice", "Sugarcane", "Pulses"],
    "Black Soil": ["Cotton", "Soybean", "Groundnut", "Maize"],
    "Clay soil": ["Paddy", "Sugarcane", "Vegetables"],
    "Red soil": ["Millets", "Groundnut", "Pulses", "Fruits"]
}

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img = Image.open(file.file).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img).logits
        _, predicted = torch.max(outputs, 1)
    
    soil_type = class_labels[predicted.item()]
    crops = crop_suggestions[soil_type]

    return {"Soil Type": soil_type, "Suitable Crops": crops}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)