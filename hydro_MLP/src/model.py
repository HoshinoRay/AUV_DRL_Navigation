import torch
import torch.nn as nn


class DeepHydroMLP(nn.Module):
    """
    MLP that maps 6-DOF velocity + acceleration (12 inputs) to 6-DOF
    hydrodynamic force/torque (6 outputs).

    Architecture: 12 -> 128 -> 256 -> 128 -> 6  (bottleneck hourglass)
    Activation: LeakyReLU(0.1) to avoid dead neurons.
    """

    def __init__(self, input_dim=12, output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)
