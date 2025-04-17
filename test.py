import torch

# 替換為你的模型路徑
model_path = "./models/fer2013_model.pth"
checkpoint = torch.load(model_path, map_location='cpu')  # 使用 CPU 載入以避免 GPU 問題

print("Checkpoint type:", type(checkpoint))
print("Checkpoint keys:", checkpoint.keys() if isinstance(
    checkpoint, dict) else "No keys (likely model instance)")
