import cv2 as cv
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple

def create_model(version: str, num_classes: int):

    if version == "v1":
        from .v1 import SmallCardNet, transform, model_path
        return SmallCardNet(num_classes), transform, model_path
    else:
        raise ValueError(f"Unknown model version: {version}")

class CardPredictor:
    def __init__(
        self,
        version: str,
        classes: List[str]
    ):
        self.classes = classes
        self._model, self._tfm, self._model_path = create_model(version, len(classes))
        self._state_dict = torch.load(self._model_path)
        self._model.load_state_dict(self._state_dict)
        self._model.eval()

    def predict_card(
        self,
        img: np.ndarray,
    ) -> Tuple[str, int, torch.Tensor]:

        """
        Run a single card image through the model and return:
        - predicted class name
        - predicted class index
        - raw logits tensor (1, num_classes)

        Args:
            img: Image as a NumPy array (H, W, 3). Assumed BGR if from OpenCV.
        """

        # Convert BGR (OpenCV) -> RGB
        if img.shape[-1] == 3:
            img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        else:
            img_rgb = img  # just in case it's already RGB/gray

        pil_img = Image.fromarray(img_rgb)

        # Apply transforms and add batch dimension
        x = self._tfm(pil_img).unsqueeze(0)  # shape: (1, C, H, W)

        # Move to same device as model
        device = next(self._model.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            logits = self._model(x)  # (1, num_classes)
            pred_idx = logits.argmax(dim=1).item()

        pred_class = self.classes[pred_idx]
        return pred_class, pred_idx, logits

