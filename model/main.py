import argparse
import torch
from torch import nn
from datetime import datetime
import os


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 3)

    @torch.jit.export
    def forward_fc1(self, x):
        x = torch.relu(self.fc1(x))

        return x

    @torch.jit.export
    def forward_fc2(self, x):
        x = torch.relu(self.fc2(x))

        return x

    @torch.jit.export
    def forward_fc3(self, x):
        x = self.fc3(x)

        return x

    def forward(self, x):
        x = self.forward_fc1(x)
        x = self.forward_fc2(x)
        x = self.forward_fc3(x)

        return x


def test():
    model = MyModel()

    x = torch.randn(1, 10)
    y: torch.Tensor = model(x)

    y1 = model.forward_fc1(x)
    y2 = model.forward_fc2(y1)
    y3 = model.forward_fc3(y2)

    print(f'{y1=}')
    print(f'{y2=}')
    print(f'{y3=}')
    print(f'{y=}')


def export():
    model = MyModel()

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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--export', default=False,
                        action=argparse.BooleanOptionalAction, help='Build and save the model')
    args = parser.parse_args()

    if not args.export:
        test()
        return
    else:
        export()


if __name__ == '__main__':
    main()
