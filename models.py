import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from custom_layers import SparseLayer, SimpleSparseLayer, NearestNeighborSparseLayer, SelfConnectedSparseLayer

class LinearNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 8)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

class DenseModel(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=None) -> None:
        super(DenseModel, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.device = device

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)

    def forward(self, input_tensor: Tensor) -> Tensor:
        flatten_tensor = self.flatten(input_tensor)
        fc1_out = self.fc1(flatten_tensor)
        fc2_out = self.fc2(fc1_out)
        fc3_out = self.fc3(fc2_out)
        return fc3_out


class CNNModel(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=None) -> None:
        super(CNNModel, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.device = device

        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 8)


    def forward(self, input_tensor: Tensor) -> Tensor:
        conv1_out = self.conv1(input_tensor)
        relu_out = F.relu(conv1_out)
        pool_out = self.pool(relu_out)

        conv2_out = self.conv2(pool_out)
        relu_out = F.relu(conv2_out)
        pool_out = self.pool(relu_out)

        flatten_tensor = self.flatten(pool_out)
        x = F.relu(self.fc1(flatten_tensor))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)

        return out

class SparseModel(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=None) -> None:
        super(SparseModel, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.device = device

        self.flatten = nn.Flatten()
        self.sl1 = nn.Linear(in_features, hidden_features)
        self.sl2 = SimpleSparseLayer(hidden_features, hidden_features, sparsity=0.5, bias=True)
        self.sl3 = nn.Linear(hidden_features, out_features)
        self.sl1 = SimpleSparseLayer(in_features, hidden_features, sparsity=0.5, bias=True)
        self.sl2 = SimpleSparseLayer(hidden_features, hidden_features, sparsity=0.5, bias=True)
        self.sl3 = SimpleSparseLayer(hidden_features, out_features, sparsity=0.5, bias=True)

    def forward(self, input_tensor: Tensor) -> Tensor:
        flatten_tensor = self.flatten(input_tensor)
        sl1_out = self.sl1(flatten_tensor)
        sl2_out = self.sl2(sl1_out)
        sl3_out = self.sl3(sl2_out)
        return sl3_out


class SparseNNModel(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=None) -> None:
        super(SparseNNModel, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.device = device

        self.flatten = nn.Flatten()
        self.sl1 = nn.Linear(in_features, hidden_features)
        self.sl2 = SimpleSparseLayer(hidden_features, hidden_features, sparsity=0.5, bias=True)
        self.sl3 = nn.Linear(hidden_features, out_features)
        self.sl1 = SimpleSparseLayer(in_features, hidden_features, sparsity=0.5, bias=True)
        self.sl2 = NearestNeighborSparseLayer(hidden_features, hidden_features, sparsity=0.5, bias=True)
        self.sl3 = SimpleSparseLayer(hidden_features, out_features, sparsity=0.5, bias=True)

    def forward(self, input_tensor: Tensor) -> Tensor:
        flatten_tensor = self.flatten(input_tensor)
        sl1_out = self.sl1(flatten_tensor)
        sl2_out = self.sl2(sl1_out)
        sl3_out = self.sl3(sl2_out)
        return sl3_out

class SelfConnectedSparseModel(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=None) -> None:
        super(SelfConnectedSparseModel, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.bias = bias
        self.device = device

        self.flatten = nn.Flatten()
        self.sl1 = nn.Linear(in_features, hidden_features)
        self.sl2 = SimpleSparseLayer(hidden_features, hidden_features, sparsity=0.5, bias=True)
        self.sl3 = nn.Linear(hidden_features, out_features)
        self.sl1 = SimpleSparseLayer(in_features, hidden_features, sparsity=0.5, bias=True)
        self.sl2 = NearestNeighborSparseLayer(hidden_features, hidden_features, sparsity=0.5, bias=True)
        self.sl3 = SelfConnectedSparseLayer(hidden_features, hidden_features, sparsity=0.5, bias=True)
        self.sl4 = SimpleSparseLayer(hidden_features, out_features, sparsity=0.5, bias=True)

    def forward(self, input_tensor: Tensor) -> Tensor:
        flatten_tensor = self.flatten(input_tensor)
        sl1_out = self.sl1(flatten_tensor)
        sl2_out = self.sl2(sl1_out)
        sl3_out = self.sl3(sl2_out)
        sl4_out = self.sl3(sl3_out)

        return sl4_out



