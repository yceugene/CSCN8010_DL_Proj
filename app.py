from ResEmoteNet.approach.ResEmoteNet import ResEmoteNet
import os
import torch
import cv2
import numpy as np
import base64
from torchvision import transforms
import sys
from flask import Flask, request, render_template, jsonify

# Add ResEmoteNet path
sys.path.append(os.path.join(os.getcwd(), "ResEmoteNet"))

app = Flask(__name__)

# Load OpenCV Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load ResEmoteNet model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResEmoteNet()
# Ensure the path is correct
model_path = "./models/best_model_fer2013_ResEmoteNet.pth"
try:
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    print("Checkpoint keys:", checkpoint.keys())
    state_dict = checkpoint['model_state_dict']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Define data preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# FER2013 class names
class_names = ['angry', 'disgust', 'fear',
               'happy', 'sad', 'surprise', 'neutral']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Receive base64-encoded image from the frontend
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Store face locations and emotions
        results = []
        for (x, y, w, h) in faces:
            # Crop the face region
            face_img = img[y:y+h, x:x+w]
            face_img_gray = cv2.cvtColor(
                face_img, cv2.COLOR_BGR2GRAY)  # Convert to RGB

            # Preprocess and perform emotion prediction
            img_tensor = transform(face_img_gray).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, preds = torch.max(outputs, 1)
                emotion = class_names[preds.item()]

            # Store face location and emotion
            results.append({
                'face': [int(x), int(y), int(w), int(h)],
                'emotion': emotion
            })

            # Draw face bounding box and emotion label (directly on the original image)
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # Encode the processed image as base64 and return it to the frontend
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'results': results, 'image': f'data:image/jpeg;base64,{img_base64}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Ensure environment variables are loaded correctly
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.2/lib64' + \
        os.environ.get('LD_LIBRARY_PATH', '')
    print("Num GPUs Available: ", torch.cuda.is_available())
    app.run(debug=True, host='0.0.0.0', port=5000)
