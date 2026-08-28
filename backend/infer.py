import io, base64, cv2, torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = {0: 'NORMAL', 1: 'BACTERIAL', 2: 'VIRAL'}

def create_model():
    m = models.densenet121(weights=None)
    in_features = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 3)
    )
    return m

model = create_model()
ckpt_path = 'models/densenet121_xray_best.pt'
ckpt = torch.load(ckpt_path, map_location=device)

# استخراج مصفوفة الأوزان الصحيحة
if isinstance(ckpt, dict):
    if 'model_state' in ckpt:
        state = ckpt['model_state']
    elif 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
else:
    state = ckpt

model.load_state_dict(state)
model.to(device)
model.eval()

# تجهيز Grad-CAM Hooks
gradients, activations = [], []
def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])
def forward_hook(module, input, output):
    activations.append(output)

target_layer = model.features.denseblock4.denselayer16.conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

def apply_clahe(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

def predict_and_cam(image_bytes: bytes):
    gradients.clear(); activations.clear()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    np_img = np.array(pil_img)
    enhanced = apply_clahe(np_img)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(enhanced).unsqueeze(0).to(device)
    
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    
    model.zero_grad()
    score = logits[0, pred_idx]
    score.backward()
    
    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (np_img.shape[1], np_img.shape[0]))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = np.uint8(0.6 * np_img + 0.4 * heatmap)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cam_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
    
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(3)},
        "gradcam_image": cam_base64
    }
