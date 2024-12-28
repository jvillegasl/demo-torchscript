import torch
from torch import nn
from datetime import datetime
import os


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 3)

    def forward(self, x):
        return self.fc(x)


if __name__ == '__main__':
    model = MyModel()

    x = torch.randn(1, 10)
    y: torch.Tensor = model(x)

    sm = torch.jit.script(model)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f'model_{timestamp}.pt'

    directory = os.path.dirname(os.path.abspath(__file__))
    directory = os.path.join(directory, 'build')

    os.makedirs(directory, exist_ok=True)

    path = os.path.join(directory, filename)

    sm.save(path)

    path = path.replace('\\', '\\\\')

    print(path, end='')
