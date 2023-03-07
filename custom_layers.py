import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseLayer(nn.Module):
    def __init__(self, in_features, out_features, sparsity, connection_ratio, bias=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=None):
        super(SparseLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.connection_ratio = connection_ratio
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)).to(device)
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.device is not None:
            self.weight = self.weight.to(self.device)
            if self.bias is not None:
                self.bias = self.bias.to(self.device)

        if torch.cuda.is_available():
            if self.device is None:
                device = torch.device('cuda')
            else:
                device = self.device

            self.weight = self.weight.to(device)
            if self.bias is not None:
                self.bias = self.bias.to(device)

        mask = torch.rand((self.out_features, self.in_features), dtype=self.dtype)
        mask = mask < self.sparsity
        num_connections = int(self.in_features * self.out_features * self.connection_ratio)
        flattened_mask = mask.flatten()
        indices = torch.argsort(flattened_mask, descending=True)[:num_connections]
        mask = torch.zeros_like(flattened_mask)
        mask[indices] = 1
        mask = mask.reshape((self.out_features, self.in_features))

        weight = self.weight * mask.float().to(device)
        output = F.linear(input, weight, self.bias)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, sparsity={}, connection_ratio={}, device={}, dtype={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.sparsity, self.connection_ratio, self.device, self.dtype
        )

class SimpleSparseLayer(nn.Module):
    def __init__(self, in_features, out_features, sparsity=0.5, bias=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super(SimpleSparseLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.use_bias = bias
        self.device = device

        # Berechnen der maximalen Anzahl an Verbindungen zwischen den Eingangs- und Ausgangsneuronen.
        max_connections = int(in_features * out_features * sparsity)

        # Erstellen eines 2D-Tensors, der alle möglichen Verbindungen zwischen den Eingangs- und Ausgangsneuronen enthält.
        connections = torch.zeros(in_features, out_features)

        # Zufälliges Auswählen von max_connections Verbindungen und Markieren als aktiv.
        active_indices = torch.randperm(in_features * out_features)[:max_connections]
        connections.view(-1)[active_indices] = 1

        # Speichern der Verbindungen als PyTorch Parameter.
        self.connections = nn.Parameter(connections, requires_grad=False)

        # Erstellen der Gewichtungsmatrix als PyTorch Parameter.
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).to(device), requires_grad=True)

        # Initialisierung der Gewichte mit Xavier-Initialisierung.
        nn.init.xavier_uniform_(self.weight)

        # Optional: Erstellen des Bias-Parameters.
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).to(device), requires_grad=True)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        # Anwenden der Verbindungen auf die Eingabe.
        x = x.matmul(self.connections * self.weight.t())

        # Optional: Hinzufügen des Biases.
        if self.use_bias:
            x = x + self.bias

        return x

class SelfConnectedSparseLayer(nn.Module):
    def __init__(self, in_features, out_features, sparsity, connection_ratio, bias=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), dtype=None):
        super(SelfConnectedSparseLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.connection_ratio = connection_ratio
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)).to(device)
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        if self.device is not None:
            self.weight = self.weight.to(self.device)
            if self.bias is not None:
                self.bias = self.bias.to(self.device)

        if torch.cuda.is_available():
            if self.device is None:
                device = torch.device('cuda')
            else:
                device = self.device

            self.weight = self.weight.to(device)
            if self.bias is not None:
                self.bias = self.bias.to(device)

        mask = torch.zeros((self.out_features, self.in_features), dtype=self.dtype)
        for i in range(self.out_features):
            # Connect to self and neighboring neurons
            start = max(0, i - 1)
            end = min(self.out_features, i + 2)
            connected_neurons = list(range(start, end))
            connected_neurons.remove(i)
            connected_neurons = [n % self.out_features for n in connected_neurons]
            connected_neurons.sort()

            # Randomly select connections based on sparsity
            num_connections = int(len(connected_neurons) * self.in_features * self.connection_ratio)
            indices = torch.randperm(len(connected_neurons) * self.in_features)[:num_connections]
            row_indices = torch.full((len(indices),), i, dtype=torch.long)
            col_indices = torch.zeros((len(indices),), dtype=torch.long)
            for j, idx in enumerate(indices):
                col_indices[j] = connected_neurons[idx // self.in_features] * self.in_features + idx % self.in_features

            # Set mask
            mask[row_indices, col_indices] = 1

        weight = self.weight * mask.float().to(device)
        output = F.linear(input, weight, self.bias)

        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, sparsity={}, connection_ratio={}, device={}, dtype={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.sparsity, self.connection_ratio, self.device, self.dtype
        )