from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field
from typing import List
import numpy as np
from PIL import Image
import torch

# Initialize FastAPI app
app = FastAPI()

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path="./weights/yolov5s.pt")


class Detection(BaseModel):
    class_name: str = Field(..., alias='class')
    x: float
    y: float
    w: float
    h: float

# Response model


class PredictionResponse(BaseModel):
    predictions: List[Detection]


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Read image
    image = Image.open(file.file)
    image = np.array(image)

    # Run inference
    results = model(image)

    # Parse results
    detections = []
    for result in results.xywh[0].cpu().numpy():
        x, y, w, h, conf, class_id = result
        class_name = model.names[int(class_id)]
        detection = {
            "class": class_name,
            "x": round(float(x), 2),
            "y": round(float(y), 2),
            "w": round(float(w), 2),
            "h": round(float(h), 2)
        }
        detections.append(detection)

    return PredictionResponse(predictions=detections)

# Run the server with: uvicorn main:app --reload
