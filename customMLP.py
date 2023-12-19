import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CustomMLP(nn.Module):
    def __init__(self, layer_sizes, layer_types):
        super(CustomMLP, self).__init__()
        self.layers = nn.ModuleList()
        for i, layer_type in enumerate(layer_types):
            if layer_type == 'tanh':
                self.layers.append(
                    TanhLayer(layer_sizes[i], layer_sizes[i + 1]))
            elif layer_type == 'relu':
                self.layers.append(
                    ReLULayer(layer_sizes[i], layer_sizes[i + 1]))
            elif layer_type == 'lstm':
                self.layers.append(
                    LSTMLayer(layer_sizes[i], layer_sizes[i + 1]))

        self.layers.append(Linear(layer_sizes[-2], layer_sizes[-1]))
        print("self.layers:", self.layers)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def get_weights(self):
        w = []
        for layer in self.layers:
            if isinstance(layer, TanhLayer) or isinstance(layer, ReLULayer):
                layer.w.data = torch.randn_like(layer.w) * np.sqrt(2 / (layer.w.size(0) + layer.w.size(1)))
                layer.b.data = torch.zeros_like(layer.b)

            w.extend(layer.w.flatten().detach().numpy())
            w.extend(layer.b.flatten().detach().numpy())

        w = np.array(w)

        return w, self

    def backpropagate(self, x):
        activations = [x]

        for layer in self.layers:
            print("layer:", layer)
            print("x:", x)
            x = layer(x)
            print("x:", x)

            activations.append(x)

        y = activations[-1] 
        print("y:", y)
        tensorList = []
        DrannDw = []
        print("activations:", activations)

        for i in reversed(range(len(self.layers))):
            print("----------------------------START OF FOR LOOP-----------------------------")
            print("activations[i]:", activations[i])
            h1 = activations[i]
            h1l = h1
            print("h1:", h1)
            print("h1l:", h1l)

            # Correctly calculate output_size for the current layer
            print(self.layers)
            print(self.layers[i].w.shape)

            output_size = self.layers[i].w.shape[0]
            # 7 > 5

            # Resize DrannDanninp for the current layer
            DrannDanninp = torch.eye(output_size)

            h1l_reshaped = h1l.t()

            print("DrannDanninp:", DrannDanninp.shape)
            print("self.layers[i].w:", self.layers[i].w)
            print("h1l shape:", h1l.shape)
            print("h1l_reshaped shape:", h1l_reshaped.shape)
            print("output_size:", output_size)
            print("h1l_reshaped repeated shape:", h1l_reshaped.repeat(output_size,1).shape)
            print("Matrix multiplication result shape:", torch.mm(DrannDanninp,self.layers[i].w))
            print("Matrix multiplication result shape:", torch.mm(DrannDanninp,self.layers[i].w).shape)
            print("output_size:", output_size)

            A1 = -(torch.mm(DrannDanninp,self.layers[i].w) * h1l_reshaped.repeat(output_size, 1))

            print("A1:", A1)
            print("A1 shape:", A1.shape)

            h1l_reshaped = torch.cat((h1l_reshaped, torch.tensor([[1]])), dim=1)

            layer_dydw = torch.kron(h1l_reshaped, A1)

            print("layer_dydw:", layer_dydw)
            print("layer_dydw shape:", layer_dydw.shape)

            print("self.layers[i].w:", self.layers[i].w)
            DrannDanninp = torch.mm(self.layers[i].w, A1.t())

            print("DrannDanninp:", DrannDanninp)
            print("DrannDanninp shape:", DrannDanninp.shape)

            tensorList.append(layer_dydw)

        print("tensorList:", tensorList)
        DrannDw = tensorList
        print("DrannDw:", DrannDw)
        print("DrannDanninp:", DrannDanninp)
        print("DrannDanninp shape:", DrannDanninp.shape)
        print("y:", y)
        print("y shape:", y.shape)

        return y, DrannDanninp, DrannDw

class TanhLayer(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super(TanhLayer, self).__init__()
        self.w = nn.Parameter(torch.randn(output_size, input_size))
        self.b = nn.Parameter(torch.randn(output_size, 1))

    def forward(self, x):
        print("self.w:", self.w)
        print("x:", x)        
        return torch.tanh(torch.mm(self.w, x) + self.b)

    def derivative(self, x):
        return 1 - torch.tanh(torch.mm(self.w, x) + self.b) ** 2


class ReLULayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(ReLULayer, self).__init__()
        self.w = nn.Parameter(torch.randn(output_size, input_size))
        self.b = nn.Parameter(torch.randn(output_size, 1))

    def forward(self, x):
        xin = torch.mm(self.w, x) + self.b
        return F.leaky_relu(xin, negative_slope=0.003)

    def derivative(self, x):
        xin = torch.mm(self.w, x) + self.b
        return torch.where(xin > 0, torch.ones_like(xin), torch.zeros_like(xin))


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.wf = nn.Parameter(torch.randn(hidden_size, input_size))
        self.wrf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bfor = nn.Parameter(torch.randn(hidden_size, 1))
        self.win = nn.Parameter(torch.randn(hidden_size, input_size))
        self.wrin = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bin = nn.Parameter(torch.randn(hidden_size, 1))
        self.wbl = nn.Parameter(torch.randn(hidden_size, input_size))
        self.wrbl = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bprev = nn.Parameter(torch.randn(hidden_size, 1))
        self.wout = nn.Parameter(torch.randn(hidden_size, input_size))
        self.wrout = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bout = nn.Parameter(torch.randn(hidden_size, 1))
        self.cprev = torch.zeros(hidden_size, 1)
        self.yprev = torch.zeros(hidden_size, 1)

    def forward(self, x):
        fg = torch.sigmoid(self.wf @ x + self.wrf @ self.yprev + self.bfor)
        ing = torch.sigmoid(self.win @ x + self.wrin @ self.yprev + self.bin)
        blg = torch.tanh(self.wbl @ x + self.wrbl @ self.yprev + self.bprev)
        outg = torch.sigmoid(self.wout @ x + self.wrout @
                             self.yprev + self.bout)
        c = blg * ing + self.cprev * fg
        y = outg * torch.tanh(c)
        self.cprev = c
        self.yprev = y
        return y


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.randn(output_size, input_size))
        self.b = nn.Parameter(torch.randn(output_size, 1))

    def forward(self, x):
        return torch.mm(self.w, x) + self.b

    def derivative(self, x):
        return torch.ones_like(x)