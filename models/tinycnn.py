import torch.nn as nn

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        z = self.conv(x).view(x.size(0), -1)
        return self.fc(z)