from ResEmoteNet.approach.ResEmoteNet import ResEmoteNet
import os
import torch
import cv2
import numpy as np
import base64
from torchvision import transforms
import sys
from flask import Flask, request, render_template, jsonify

# 添加 ResEmoteNet 路徑
sys.path.append(os.path.join(os.getcwd(), "ResEmoteNet"))

app = Flask(__name__)

# 載入 OpenCV 的 Haar Cascade 人臉檢測器
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 載入 ResEmoteNet 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResEmoteNet()
model_path = "./models/fer2013_model.pth"  # 確保路徑正確
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

# 定義數據預處理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# FER2013 類別名稱
class_names = ['angry', 'disgust', 'fear',
               'happy', 'sad', 'surprise', 'neutral']


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    try:
        # 從前端接收 base64 編碼的圖像
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # 轉為灰度圖像進行人臉檢測
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # 儲存人臉位置和情緒
        results = []
        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # 轉為 RGB

            # 預處理並進行推理
            img_tensor = transform(face_img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, preds = torch.max(outputs, 1)
                emotion = class_names[preds.item()]

            # 繪製人臉框和情緒標籤（直接在圖像上）
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img, emotion, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            results.append({
                'face': [int(x), int(y), int(w), int(h)],
                'emotion': emotion
            })

        # 將處理後的圖像編碼為 base64，返回給前端
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'results': results, 'image': f'data:image/jpeg;base64,{img_base64}'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 確保環境變數正確載入
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.2/lib64' + \
        os.environ.get('LD_LIBRARY_PATH', '')
    print("Num GPUs Available: ", torch.cuda.is_available())
    app.run(debug=True, host='0.0.0.0', port=5000)
