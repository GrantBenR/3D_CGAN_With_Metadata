from torch import nn, cat, Tensor
class Discriminator(nn.Module):
    def __init__(self, metadata_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3 + metadata_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, vectors, metadata) -> Tensor:
        x = cat((vectors, metadata), dim=1)
        return self.model(x)