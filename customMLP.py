import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random   

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


        self.scale_weights(scaling_factor=0.01)


    def forward(self, x):
        for layer in self.layers:
            x = x.to(dtype=torch.float64)

            x = layer(x)
        return x


    def scale_weights(self, scaling_factor):
        with torch.no_grad():  
            for layer in self.layers:
                for w in layer.w.data:
                    w *= scaling_factor * random.uniform(0.9, 1.1)
                layer.b.data *= scaling_factor

    
    def reinitialize_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'w'):
                if isinstance(layer, (TanhLayer, Linear)): 
                    nn.init.xavier_uniform_(layer.w)
                elif isinstance(layer, ReLULayer): 
                    nn.init.kaiming_uniform_(layer.w, mode='fan_in', nonlinearity='relu')

            if hasattr(layer, 'b'):
                nn.init.constant_(layer.b, 0)

        self.scale_weights(scaling_factor=0.0001)

        weights = []
        for layer in self.layers:
            if hasattr(layer, 'w') and hasattr(layer, 'b'):
                weights.append(layer.w.data.cpu().numpy().flatten())
                weights.append(layer.b.data.cpu().numpy().flatten())

        weights = np.concatenate(weights)
        return weights, self    


    def set_weights(self, new_weights):
        start = 0
        for layer in self.layers:
            if hasattr(layer, 'w') and hasattr(layer, 'b'):
                weight_num_elements = torch.numel(layer.w)
                bias_num_elements = torch.numel(layer.b)

                new_w = new_weights[start:start+weight_num_elements]
                new_b = new_weights[start+weight_num_elements:start+weight_num_elements+bias_num_elements]

                start += weight_num_elements + bias_num_elements

                new_w_tensor = torch.from_numpy(new_w).view_as(layer.w).type_as(layer.w)
                new_b_tensor = torch.from_numpy(new_b).view_as(layer.b).type_as(layer.b)

                layer.w.data = new_w_tensor
                layer.b.data = new_b_tensor
        
        '''
        for layer in self.layers:
            print("weights: ", layer.w)
            print("biases: ", layer.b)
            if layer.w.shape == torch.Size([3, 3]):
                with torch.no_grad():
                    transposed_w = layer.w.t().clone() 
                    layer.w.copy_(transposed_w)  
        '''




    def print_weights_and_biases(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'w') and hasattr(layer, 'b'):
                print(f"Layer {i} weights (w): \n{layer.w.data}")
                print(f"Layer {i} biases (b): \n{layer.b.data}\n")


    def get_weights(self):
        weights = []
        for layer in self.layers:
            w = layer.w.data.cpu().numpy().flatten()
            b = layer.b.data.cpu().numpy().flatten()
            weights.extend(w)
            weights.extend(b)
        return np.array(weights), self

    def backpropagate(self, x, ny):
        activations = [x]

        for layer in self.layers:

            x = x.to(dtype=torch.float64)
            x = layer(x)

            activations.append(x)
            
        # y = output
        y = activations[-1]

        # x = input
        x = activations[0]

        # h1 
        tensorList = []
        DrannDw = []
        output_size = self.layers[-1].w.shape[0]
        DrannDanninp = torch.eye(output_size, dtype=torch.float64)
        A1 = DrannDanninp
        tensor_size = 0
        '''
        with open("back.txt", "a") as f:
            f.write("activations\n")
            f.write(str(activations))
            f.write("\n")
            f.write("y\n")
            f.write(str(y))
            f.write("\n")
            f.write("x\n")
            f.write(str(x))
            f.write("\n")
            f.write("")
            f.write(str(range(len(self.layers))))
            f.write("\n")
        '''
    
        for i in reversed(range(len(self.layers))):

            h1 = activations[i]
            h1l = self.layers[i-1].derivative(h1)
            h1l_reshaped = h1l.t()        

            '''
            1º valor i self.layers -1
               A2 =  matriz identidade tamanho = n de outputs

            2º valor i self.layers 
                obter h1l  =  self.layers[i].derivative(h1)
                A1 = -(torch.mm(A2, self.layers[i].w) * h1l.repeat(output_size, 1))

            3º valor i self.layers
                dydw = matriz em linha kron([input,1],A1).cat(kron([h1,1],A2)) 
                dydx = A1 * self.layers[i].w

            '''
            
            h1_reshaped = torch.cat((h1.t(), torch.tensor([[1]])), dim=1)
            
            layer_dydw = torch.kron(h1_reshaped,A1)

            tensor_size = tensor_size + layer_dydw.shape[1] 
            tensorList.insert(0, layer_dydw)


            if i == 0:
                break

            A1 = -(torch.mm(DrannDanninp,self.layers[i].w) * h1l_reshaped.repeat(output_size, 1))


            DrannDanninp = A1

            h1l_reshaped = torch.cat((h1l_reshaped, torch.tensor([[1]])), dim=1)


        DrannDanninp = torch.mm(A1,self.layers[0].w)


        DrannDw = tensorList

        DrannDw = torch.cat(DrannDw, dim=1)
        DrannDw = DrannDw.view(ny, tensor_size)


        return y, DrannDanninp, DrannDw

class TanhLayer(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super(TanhLayer, self).__init__()
        self.w = nn.Parameter(torch.Tensor(output_size, input_size).double())
        self.b = nn.Parameter(torch.Tensor(output_size, 1).double())
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b)

    def forward(self, x):
        return torch.tanh(torch.mm(self.w, x) + self.b)

    def derivative(self, x):
        return (x ** 2) -1 


class ReLULayer(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super(ReLULayer, self).__init__()
        self.w = nn.Parameter(torch.Tensor(output_size, input_size).double())
        self.b = nn.Parameter(torch.Tensor(output_size, 1).double())
        nn.init.kaiming_uniform(self.w)
        nn.init.zeros_(self.b)

    def forward(self, x):
        xin = torch.mm(self.w, x) + self.b
        return F.relu(xin)

    def derivative(self, x):
        return (x > 0).double()


class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(LSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=batch_first)
        
        self.hidden = None
        self.cell = None

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x, (self.hidden, self.cell))

        return output

    def reset_state(self):
        self.hidden = None
        self.cell = None


class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.Tensor(output_size, input_size).double())
        self.b = nn.Parameter(torch.Tensor(output_size, 1).double())
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b) 

    def forward(self, x):
        return torch.mm(self.w, x) + self.b

    def derivative(self, x):
        return torch.ones_like(x) 