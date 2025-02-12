from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
from torchvision import transforms
from transformers import ViTForImageClassification, TrOCRProcessor, VisionEncoderDecoderModel
import io
import os
from PIL import Image

app = Flask(__name__)

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

current_dir = os.path.dirname(os.path.abspath(__file__))
color_model_path = os.path.join(current_dir, "best_model.pth")
ocr_model_path = os.path.join(current_dir, "best.onnx")

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo_model.classes = [2, 5, 7]

num_classes = 15
color_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)
color_model.load_state_dict(torch.load(color_model_path, map_location=torch.device('cpu')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
color_model = color_model.to(device)
color_model.eval()

ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

color_labels = ['beige', 'black', 'blue', 'brown', 'gold', 'green', 'grey', 'orange', 'pink', 'purple', 'red', 'silver', 'tan', 'white', 'yellow']

def preprocess_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def predict_color(model, image):
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = data_transforms(Image.fromarray(image)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor).logits
        _, preds = torch.max(outputs, 1)
    return color_labels[preds[0]]

def load_ocr_model(onnx_model_path):
    net = cv2.dnn.readNetFromONNX(onnx_model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def get_detections(img, net):
    image = img.copy()
    row, col, d = image.shape
    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3), dtype=np.uint8)
    input_image[0:row, 0:col] = image

    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    detections = preds[0]

    return input_image, detections

def non_maximum_suppression(input_image, detections, conf_threshold=0.4, nms_threshold=0.45):
    boxes = []
    confidences = []
    image_w, image_h = input_image.shape[:2]
    x_factor = image_w / INPUT_WIDTH
    y_factor = image_h / INPUT_HEIGHT

    for i in range(len(detections)):
        row = detections[i]
        confidence = row[4]
        if confidence > conf_threshold:
            class_score = row[5]
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                left = int((cx - 0.5 * w) * x_factor)
                top = int((cy - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                confidences.append(float(confidence))
                boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    index = cv2.dnn.NMSBoxes(boxes_np, confidences_np, conf_threshold, nms_threshold)

    return boxes_np, confidences_np, index

def extract_text_with_trocr(image):
    pixel_values = ocr_processor(images=image, return_tensors="pt").pixel_values
    output_ids = ocr_model.generate(pixel_values)
    text = ocr_processor.batch_decode(output_ids, skip_special_tokens=True)
    return text[0]

def detect_and_classify_cars(image):
    preprocessed_image = preprocess_image(image)
    results = yolo_model(preprocessed_image)

    largest_box = None
    max_area = 0

    for *box, conf, cls in results.xyxy[0]:
        if conf > 0.3:
            x1, y1, x2, y2 = map(int, box)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                largest_box = (x1, y1, x2, y2)

    if largest_box:
        x1, y1, x2, y2 = largest_box
        car_image = preprocessed_image[y1:y2, x1:x2]

        color = predict_color(color_model, car_image)

        ocr_net = load_ocr_model(ocr_model_path)
        input_image, detections = get_detections(car_image, ocr_net)
        boxes_np, confidences_np, index = non_maximum_suppression(input_image, detections)

        if len(boxes_np) > 0:
            x, y, w, h = boxes_np[0]
            plate_img = car_image[y:y+h, x:x+w]
            plate_text = extract_text_with_trocr(plate_img)
        else:
            plate_text = "No plate detected"

        return color, plate_text
    
    return "No car detected", None

@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to the Car Detection and Classification API"}), 200

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            in_memory_file = io.BytesIO(file.read())
            img = np.frombuffer(in_memory_file.getbuffer(), dtype=np.uint8)
            image = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not read the image")

            color, plate_text = detect_and_classify_cars(image)
            
            result_data = {
                "car_color": color,
                "license_plate_text": plate_text
            }

            return jsonify(result_data), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": "File processing failed"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)