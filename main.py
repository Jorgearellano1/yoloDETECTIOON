import ssl
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import threading
import time

ssl._create_default_https_context = ssl._create_unverified_context

# Cargar modelos de YOLOv5 y ResNet50
detector_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
vehicle_class_ids = [2, 3, 5, 7]  # Clases para automóviles, camiones, autobuses y motocicletas

classifier_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
classifier_model.eval()

# Etiquetas de ImageNet
imagenet_labels = ["car", "truck", "bus", "motorcycle", "bicycle"]

# Preprocesamiento para ResNet
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Definir colores básicos
COLOR_NAMES = {
    "Red": (255, 0, 0), "Green": (0, 255, 0), "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0), "Black": (0, 0, 0), "White": (255, 255, 255)
}

# Base de datos simulada
tracking_db = {}
vehicles_detected = {}

def closest_color(requested_color):
    min_colors = {}
    for name, rgb in COLOR_NAMES.items():
        rd = (rgb[0] - requested_color[0]) ** 2
        gd = (rgb[1] - requested_color[1]) ** 2
        bd = (rgb[2] - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]

def get_average_color(image):
    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return closest_color(avg_color.astype(int))

def classify_vehicle(vehicle_image):
    pil_image = Image.fromarray(cv2.cvtColor(vehicle_image, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = classifier_model(input_batch)
        _, predicted_idx = torch.max(output, 1)

    return imagenet_labels[predicted_idx % len(imagenet_labels)]

# Inicia el tracking en un hilo separado
def tracking_loop(tracking_id):
    global tracking_db, vehicles_detected

    while tracking_db[tracking_id]["IsActive"]:
        for camera_id, video_path in enumerate(["video1.mp4", "video2.mp4"]):
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                continue

            results = detector_model(frame)
            for detection in results.xyxy[0].cpu().numpy():
                x_min, y_min, x_max, y_max, confidence, class_id = detection[:6]
                if int(class_id) in vehicle_class_ids and confidence > 0.5:
                    vehicle_area = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                    color_name = get_average_color(vehicle_area)
                    vehicle_type = classify_vehicle(vehicle_area)

                    if (
                        color_name == tracking_db[tracking_id]["color"] and
                        vehicle_type == tracking_db[tracking_id]["model"]
                    ):
                        vehicles_detected[camera_id] = {
                            "type": vehicle_type,
                            "color": color_name,
                            "bounding_box": [x_min, y_min, x_max, y_max]
                        }

            cap.release()
        time.sleep(1)

# Modelo para la API
class TrackingRequest(BaseModel):
    model: str
    color: str

class StopTrackingRequest(BaseModel):
    tracking_id: int

# Crear FastAPI
app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "API para detección y tracking de vehículos",
        "endpoints": [
            {"route": "/tracking", "method": "POST", "description": "Inicia el tracking de un vehículo."},
            {"route": "/tracking/stop", "method": "POST", "description": "Detiene el tracking."},
            {"route": "/vehicles", "method": "GET", "description": "Obtiene vehículos detectados de una cámara."}
        ]
    }

@app.post("/tracking")
def start_tracking(request: TrackingRequest):
    tracking_id = len(tracking_db) + 1
    tracking_db[tracking_id] = {
        "model": request.model,
        "color": request.color,
        "IsActive": True
    }
    threading.Thread(target=tracking_loop, args=(tracking_id,), daemon=True).start()
    return {"message": "Tracking iniciado", "tracking_id": tracking_id}

@app.post("/tracking/stop")
def stop_tracking(request: StopTrackingRequest):
    tracking_id = request.tracking_id
    if tracking_id not in tracking_db:
        raise HTTPException(status_code=404, detail="Tracking ID no encontrado")
    tracking_db[tracking_id]["IsActive"] = False
    return {"message": "Tracking detenido", "tracking_id": tracking_id}

@app.get("/vehicles")
def get_vehicles(camera_id: Optional[int] = 0):
    if camera_id not in vehicles_detected:
        raise HTTPException(status_code=404, detail="No se detectaron vehículos en esta cámara")
    return {"camera_id": camera_id, "vehicles": vehicles_detected[camera_id]}
