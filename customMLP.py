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


        self.scale_weights(scaling_factor=0.1)


    def forward(self, x):
        for layer in self.layers:

            x = x.to(dtype=torch.float32)

            x = layer(x)
        return x


    def scale_weights(self, scaling_factor):
        with torch.no_grad():
            for layer in self.layers:
                if hasattr(layer, 'w') and hasattr(layer, 'b'):
                    for w in layer.w.data:
                        w *= scaling_factor * torch.tensor(random.uniform(0.9, 1.1), dtype=torch.float32)
                    layer.b.data *= scaling_factor

                elif isinstance(layer, LSTMLayer):
                    for param in layer.lstm.parameters():
                        param *= scaling_factor * torch.tensor(random.uniform(0.9, 1.1), dtype=param.dtype)

    
    def reinitialize_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'w'):
                if isinstance(layer, (TanhLayer, Linear)): 
                    nn.init.xavier_uniform_(layer.w)
                elif isinstance(layer, ReLULayer): 
                    nn.init.kaiming_uniform_(layer.w, mode='fan_in', nonlinearity='relu')

            if hasattr(layer, 'b'):
                nn.init.constant_(layer.b, 0)

        self.scale_weights(scaling_factor=0.1)

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
        
            elif isinstance(layer, LSTMLayer):
                for param in layer.lstm.parameters():
                    num_params = param.numel()
                    param_data = new_weights[start:start + num_params]
                    param_tensor = torch.from_numpy(param_data).view_as(param).type_as(param)
                    param.data.copy_(param_tensor)
                    start += num_params





    def print_weights_and_biases(self):
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'w') and hasattr(layer, 'b'):
                print(f"Layer {i} weights (w): \n{layer.w.data}")
                print(f"Layer {i} biases (b): \n{layer.b.data}\n")

    '''
    def get_weights(self):
        weights = []
        for layer in self.layers:
            w = layer.w.data.cpu().numpy().flatten()
            b = layer.b.data.cpu().numpy().flatten()
            weights.extend(w)
            weights.extend(b)
        return np.array(weights), self
    '''

    def get_weights(self):
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'w') and hasattr(layer, 'b'):
                weights.append(layer.w.data.cpu().numpy().flatten())
                weights.append(layer.b.data.cpu().numpy().flatten())
            elif isinstance(layer, LSTMLayer):
                for param in layer.lstm.parameters():
                    weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights), self

    def get_weights_solo(self):
        weights = []
        for layer in self.layers:
            if hasattr(layer, 'w') and hasattr(layer, 'b'):
                weights.append(layer.w.data.cpu().numpy().flatten())
                weights.append(layer.b.data.cpu().numpy().flatten())
            elif isinstance(layer, LSTMLayer):
                for param in layer.lstm.parameters():
                    weights.append(param.data.cpu().numpy().flatten())
        return np.concatenate(weights)

    def backpropagate(self, x, ny, target=None, loss_fn=None):
        contains_lstm = any(isinstance(layer, LSTMLayer) for layer in self.layers)

        if contains_lstm:
            assert loss_fn is not None and target is not None, \
                "Loss function and target required for LSTM backpropagation."
            return self.backpropagate_lstm(x, target, loss_fn)
        else:
            activations = [x]

            for layer in self.layers:


                x = x.to(dtype=torch.float32)

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
            DrannDanninp = torch.eye(output_size, dtype=torch.float32)

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
                
                h1_reshaped = torch.cat((h1.t(), torch.tensor([[1]], dtype=torch.float32)), dim=1)
                
                layer_dydw = torch.kron(h1_reshaped,A1)

                tensor_size = tensor_size + layer_dydw.shape[1] 
                tensorList.insert(0, layer_dydw)


                if i == 0:
                    break

                A1 = -(torch.mm(DrannDanninp,self.layers[i].w) * h1l_reshaped.repeat(output_size, 1))


                DrannDanninp = A1

                h1l_reshaped = torch.cat((h1l_reshaped, torch.tensor([[1]], dtype=torch.float32)), dim=1)



            DrannDanninp = torch.mm(A1,self.layers[0].w)


            DrannDw = tensorList

            DrannDw = torch.cat(DrannDw, dim=1)
            DrannDw = DrannDw.view(ny, tensor_size)


            return y, DrannDanninp, DrannDw

    def backpropagate_lstm(self, x, target, loss_fn):
        x = x.detach().requires_grad_()
        y = self.forward(x)

        ny = y.shape[0]
        grad_matrix = []

        params = [p for p in self.parameters() if p.requires_grad]

        for i in range(ny):
            grad_outputs = torch.zeros_like(y)
            grad_outputs[i] = 1.0

            grads = torch.autograd.grad(
                outputs=y,
                inputs=params,
                grad_outputs=grad_outputs,
                retain_graph=True,
                allow_unused=True
            )

            grad_list = [g.view(-1) for g in grads if g is not None]
            grad_vector = torch.cat(grad_list)
            grad_matrix.append(grad_vector)

        DrannDw = torch.stack(grad_matrix) 

        ny = y.shape[0]
        nx = x.shape[0]
        DrannDanninp = torch.zeros(ny, nx)

        for i in range(ny):
            grads = torch.autograd.grad(
                y[i], x, retain_graph=True, allow_unused=True
            )[0]
            if grads is not None:
                DrannDanninp[i] = grads.view(-1)

        return y, DrannDanninp, DrannDw



class Linear(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super(Linear, self).__init__()
        self.w = nn.Parameter(torch.Tensor(output_size, input_size))
        self.b = nn.Parameter(torch.Tensor(output_size, 1))
        nn.init.xavier_uniform_(self.w)
        nn.init.zeros_(self.b) 

    def forward(self, x):
        return torch.mm(self.w, x) + self.b

    def derivative(self, x):
        return torch.ones_like(x) 

class TanhLayer(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super(TanhLayer, self).__init__()
        self.w = nn.Parameter(torch.Tensor(output_size, input_size))
        self.b = nn.Parameter(torch.Tensor(output_size, 1))
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
    def __init__(self, input_size, hidden_size):
        super(LSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"LSTM expects 2D input (input_size, batch_size), got {x.shape}")
        
        x = x.t().unsqueeze(0) 

        output, (hn, cn) = self.lstm(x)
        
        return hn.squeeze(0).t() 



# TODO: SiLU activation
# TODO: Implement KAN NN based on: https://arxiv.org/pdf/2404.19756