import torch
from torch import nn


ACTION_SIZE = 8


class ImageStateEncoder(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.output_size = 128

    def forward(self, target, canvas):
        return self.net(torch.cat([target, canvas], dim=1))


class PainterPolicy(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.encoder = ImageStateEncoder(image_size)
        feature_size = self.encoder.output_size + 1

        self.shared = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(128, ACTION_SIZE),
            nn.Sigmoid(),
        )
        self.log_std_head = nn.Linear(128, ACTION_SIZE)

    def forward(self, target, canvas, step_percentage):
        if step_percentage.ndim == 1:
            step_percentage = step_percentage.unsqueeze(1)

        features = self.encoder(target, canvas)
        features = torch.cat([features, step_percentage], dim=1)
        features = self.shared(features)

        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(-5.0, -2.0)
        std = log_std.exp()
        return mean, std

    def sample_action(self, target, canvas, step_percentage):
        mean, std = self(target, canvas, step_percentage)
        distribution = torch.distributions.Normal(mean, std)
        action = distribution.sample().clamp(0.0, 1.0)
        return action, mean, std


class Critic(nn.Module):
    def __init__(self, image_size):
        super().__init__()

        self.encoder = ImageStateEncoder(image_size)
        feature_size = self.encoder.output_size + 1 + ACTION_SIZE

        self.net = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, target, canvas, step_percentage, action):
        if step_percentage.ndim == 1:
            step_percentage = step_percentage.unsqueeze(1)

        features = self.encoder(target, canvas)
        features = torch.cat([features, step_percentage, action], dim=1)
        return self.net(features)
