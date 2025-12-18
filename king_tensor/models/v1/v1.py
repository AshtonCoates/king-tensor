import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from ..card_predictor import InferenceModel

_model_path = Path(__file__).parent.resolve() / './pbs_naive.pth'

_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

class _SmallCardNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)   # adjust 8×8 to your resized size
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

classes = ['bandit', 'battle_ram', 'electro_wizard', 'minions', 'pekka', 'poison', 'royal_ghost', 'zap']

class ModelV1(InferenceModel):
    def build_model(self) -> nn.Module:
        return _SmallCardNet(len(self.classes))


if __name__ == '__main__':
    ds_path = Path("data/deckshop_cards").resolve()
    model = ModelV1(_model_path, classes, _transform, _device)
