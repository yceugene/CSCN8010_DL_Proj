import torch

# Replace with your model path
model_path = "./models/fer2013_model.pth"
# Load using CPU to avoid GPU issues
checkpoint = torch.load(model_path, map_location='cpu')

print("Checkpoint type:", type(checkpoint))
print("Checkpoint keys:", checkpoint.keys() if isinstance(
    checkpoint, dict) else "No keys (likely model instance)")
