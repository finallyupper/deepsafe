import torch.nn as nn

class Adversary_Init(nn.Module):
    """Discriminator, used in L_D of eq(8)"""
    def __init__(self, in_channels):
        super(Adversary_Init, self).__init__()
        self.model = nn.Sequential(
            # First Conv-BN-ReLU block
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Second Conv-BN-ReLU block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Third Conv-BN-ReLU block
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Average pooling layer
            nn.AdaptiveAvgPool2d((1, 1)),

            # Fully connected layer with Sigmoid
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)