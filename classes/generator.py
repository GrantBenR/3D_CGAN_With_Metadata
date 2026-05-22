from torch import nn, cat, Tensor

class Generator(nn.Module):
    def __init__(self, metadata_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2 + metadata_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 3), 
        )

    def forward(self, z, metadata) -> Tensor:
        x = cat((z, metadata), dim=1)
        return self.model(x)