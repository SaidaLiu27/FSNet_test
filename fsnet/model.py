from torch import nn

class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, d_out: int = 2):
        super().__init__()

        self.net = nn.Sequential(
            
            nn.Linear(d_in, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)
