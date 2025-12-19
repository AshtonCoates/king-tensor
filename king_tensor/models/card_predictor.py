import cv2 as cv
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Tuple, Any, Callable, Mapping
from abc import ABC, abstractmethod
from pathlib import Path

class FlatImageDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        card_paths = self.root.glob('*.png')
        self.paths = [path for path in card_paths]

        # label name = filename without extension
        self.class_names = [p.stem for p in self.paths]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGBA")
        if self.transform:
            img = self.transform(img)

        label_name = path.stem
        label = self.class_to_idx[label_name]
        return img, label

class InferenceModel(ABC):

    def __init__(
        self,
        model_path: str | Path,
        classes: List[str],
        tfm: Callable[[Any], Any] = lambda x: x,
        device: torch.device | str = "cpu",
    ):
        self.model_path = Path(model_path)
        self.classes = classes
        self.tfm = tfm
        self.device = device

        self.model = self.build_model()
        self.load_weights(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Return the architecture for this version."""
        raise NotImplementedError

    def load_weights(self, path: Path) -> None:
        """Default: load a state_dict. Override if you use TorchScript, safetensors, etc."""
        state = torch.load(path, map_location="cpu")
        # handle common checkpoint formats
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.model.load_state_dict(state)

    @torch.inference_mode()
    def __call__(self, data: Any) -> Any:
        x = self.tfm(data).unsqueeze(0)
        x = self._to_device(x, self.device)
        y = self.model(x)
        return self.postprocess(y)

    def postprocess(self, y: Any) -> Any:
        pred_idx = y.argmax(dim=1).item()
        pred_class = self.classes[pred_idx]
        return pred_class, pred_idx, y

    def _to_device(self, obj: Any, device: torch.device | str) -> Any:
        if torch.is_tensor(obj):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: self._to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            t = [self._to_device(v, device) for v in obj]
            return type(obj)(t)
        return obj

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

