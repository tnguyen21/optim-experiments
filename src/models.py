import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic ResNet block with 3x3 convolutions"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity
        out = F.relu(out)

        return out


class ResNet8(nn.Module):
    """ResNet-8 architecture for CIFAR-10

    Architecture:
    - Initial conv: 3 -> 16 channels
    - Stage 1: 16 channels, 2 BasicBlocks (layers 1-2)
    - Stage 2: 32 channels, 2 BasicBlocks (layers 3-4)
    - Stage 3: 64 channels, 2 BasicBlocks (layers 5-6)
    - avgpool + fc (layer 7)

    Total: 1 (initial) + 6 (block convs) + 1 (fc) = 8 layers
    """

    def __init__(self, num_classes=10):
        super(ResNet8, self).__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Stage 1: 16 channels, 2 blocks
        self.stage1 = nn.Sequential(BasicBlock(16, 16, stride=1), BasicBlock(16, 16, stride=1))

        # Stage 2: 32 channels, 2 blocks
        self.stage2 = nn.Sequential(
            BasicBlock(16, 32, stride=2),  # Downsample
            BasicBlock(32, 32, stride=1),
        )

        # Stage 3: 64 channels, 2 blocks
        self.stage3 = nn.Sequential(
            BasicBlock(32, 64, stride=2),  # Downsample
            BasicBlock(64, 64, stride=1),
        )

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using standard ResNet initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))

        # Three stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_features(self, x):
        """Extract intermediate features for analysis (future use)"""
        features = {}

        # Initial conv
        x = F.relu(self.bn1(self.conv1(x)))
        features["initial"] = x

        # Three stages
        x = self.stage1(x)
        features["stage1"] = x

        x = self.stage2(x)
        features["stage2"] = x

        x = self.stage3(x)
        features["stage3"] = x

        # Final layers
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features["avgpool"] = x

        x = self.fc(x)
        features["output"] = x

        return x, features
