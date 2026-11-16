"""Architectures designed for this project. Constructors initialize fresh weights."""

from torch import nn


def conv_norm(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResidualBlock(nn.Module):
    """Learn a local correction alongside an identity/projected shortcut."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.first = conv_norm(cin, cout, stride)
        self.second = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.shortcut = nn.Identity() if cin == cout and stride == 1 else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.second(self.first(x)) + self.shortcut(x))


class LeafCNN(nn.Module):
    def __init__(self, architecture="plain_v1", channels=(24, 48, 96, 192),
                 num_classes=177, dropout=0.2):
        super().__init__()
        if architecture not in {"plain_v1", "residual_v1"}:
            raise ValueError(f"Unknown architecture: {architecture}")
        if num_classes != 177 or len(channels) != 4:
            raise ValueError("The final LeafCheck contract requires 177 classes and four stages.")
        if architecture == "plain_v1":
            layers = []
            cin = 3
            for cout in channels:
                layers.append(conv_norm(cin, cout, 2))
                cin = cout
        else:
            layers = [conv_norm(3, channels[0], 2)]
            cin = channels[0]
            for index, cout in enumerate(channels):
                layers.append(ResidualBlock(cin, cout, 1 if index == 0 else 2))
                cin = cout
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
                                        nn.Dropout(dropout), nn.Linear(channels[-1], num_classes))

    def forward(self, x):
        # Return logits. Cross-entropy handles its own log-softmax.
        return self.classifier(self.features(x))
