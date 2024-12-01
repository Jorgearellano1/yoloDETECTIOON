import ssl
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

ssl._create_default_https_context = ssl._create_unverified_context

print("Cargando modelos...")
detector_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
vehicle_class_ids = [2, 3, 5, 7]

classifier_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
classifier_model.eval()

imagenet_labels = ["car", "truck", "bus", "motorcycle", "bicycle"]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

COLOR_NAMES = {
    "Red": (255, 0, 0), "Green": (0, 255, 0), "Blue": (0, 0, 255),
    "Yellow": (255, 255, 0), "Black": (0, 0, 0), "White": (255, 255, 255)
}

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

def process_video(video_path, model_target=None, color_target=None):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video finalizado.")
            break

        results = detector_model(frame)
        for detection in results.xyxy[0].cpu().numpy():
            x_min, y_min, x_max, y_max, confidence, class_id = detection[:6]
            if int(class_id) in vehicle_class_ids and confidence > 0.5:
                vehicle_area = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                color_name = get_average_color(vehicle_area)
                vehicle_type = classify_vehicle(vehicle_area)

                rect_color = (255, 0, 0)  # Azul por defecto
                if model_target and color_target:
                    if color_name == color_target and vehicle_type == model_target:
                        rect_color = (0, 255, 0)
                        print(f"Vehículo detectado: {vehicle_type} - {color_name}")

                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), rect_color, 2)
                cv2.putText(
                    frame,
                    f"{vehicle_type} - {color_name}",
                    (int(x_min), int(y_min) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    rect_color,
                    2
                )

        # Mostrar el video en una ventana
        cv2.imshow("Vehicle Tracking", frame)

        # Presiona 'q' para salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

target_model = "car"
target_color = "Red"

video_file = "video.mp4"

process_video(video_file, target_model, target_color)
