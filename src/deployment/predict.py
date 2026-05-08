# src/deployment/predict.py

import torch
import torch.nn.functional as F
from torchvision import transforms

from src.utils.model_outputs import get_main_logits


def get_prediction_transform(image_size, mean, std):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


def predict_image(model, image, transform, class_names, device):
    model.eval()

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(get_main_logits(outputs), dim=1)

    confidence, pred_idx = torch.max(probabilities, dim=1)

    pred_idx = pred_idx.item()
    confidence = confidence.item()

    return {
        "predicted_index": pred_idx,
        "predicted_class": class_names[pred_idx],
        "confidence": confidence,
        "probabilities": {
            class_names[i]: probabilities[0][i].item()
            for i in range(len(class_names))
        },
        "image_tensor": image_tensor.squeeze(0).detach().cpu()
    }
