from torch.optim import Adam
from torch import nn

def CreateOptimizer(component: nn.Module, lr=0.001) -> Adam:
    return Adam(component.parameters(), lr=lr)