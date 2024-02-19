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


    def forward(self, x):
        for layer in self.layers:
            x = x.to(dtype=torch.float64)

            x = layer(x)
        return x

    def get_weights(self):
        
        w = []
        '''
        w = [1.2006380998802779E-5, 2.1145603844498155E-5, -2.9792708564293911E-5, 
            3.9705489251404438E-5, 1.3830384891204938E-5, -3.7745464810076942E-5, 
            1.8898710862674139E-5, 8.54514207552331E-5, -5.1638738811174825E-5, 
            -2.4644163223721052E-5, -2.5550442579173686E-5, 1.1649992669326621E-5, 
            -7.3530273699083236E-5, 1.3646454191249966E-6, -5.7385955676215851E-6, 
            -3.3497613529473829E-5, 5.4707061686916957E-5, -5.3170300215709959E-5, 
            3.9707141736269434E-5, 0.00011711086081068987, -7.1751924436230878E-5, 
            -8.7424343205724235E-6, -1.2793632441151076E-5, 6.0444295801089731E-5, 
            -5.255092438286284E-5, 5.0953842348647721E-5, -4.1125752593374134E-6, 
            -5.1816807549343639E-6, 4.5700344435327305E-5, 6.5181786475404291E-5, 
            -8.3270066681982185E-6, -8.4601686111919983E-6, -4.1418435541307076E-5, 
            9.5554792377369282E-6, 1.4959000374078771E-5, 6.3696624477199682E-5, 
            4.9079581073062922E-6, -3.5665949760420265E-5, -2.4075706267310215E-5, 
            6.2302037908920361E-7, -6.44732277504166E-5, 5.4933190189396371E-6, 
            6.1999655688974784E-6, -1.3988427097698258E-5, 5.6401763585572439E-7, 
            6.4057947712191018E-5, -6.881686585158234E-5, -5.4274710006251171E-6, 
            8.0881509726190815E-6, 2.8262927089706112E-5, -3.99489319675245E-5, 
            -6.8394363035372322E-5, 2.3956123454871972E-5, 5.5313416137667989E-5, 
            -2.4440276243754793E-5, 3.8356884120287632E-9, -8.6271856230424011E-9, 
            -8.2971798692169455E-10, 3.3600077232850862E-9, 1.7986248352576827E-11, 
            1.1660650421754941E-5, 3.3330963701783143E-5, -0.00011243958129051766, 
            -2.6301330385292157E-5, -1.0651583023857215E-5, -2.5398505408298288E-5, 
            -5.1712164290481639E-5, 5.4242317487592179E-5, -9.7505245522709418E-5, 
            6.4388951326564055E-5, 2.1908858472464974E-5, -4.4166336403409615E-5, 
            -1.3023311223859764E-5, 4.2180940684206168E-5, -2.3556997420955107E-5, 
            5.4952569228907227E-5, 5.9031204133111162E-5, -8.0201551147802521E-7, 
            1.6108975213004943E-5, 1.027236329492165E-5, 8.50061867211995E-6, 
            -5.6218226119333933E-5, -6.4337170828920051E-5, -5.1403119669884979E-5, 
            -3.1542718154199429E-5, -1.2118805152745707E-6, 6.0516872069085064E-5, 
            -1.6573544932992107E-5, -5.1250843520096192E-6, -8.0994334557411835E-5, 
            -7.7688887085433686E-6, 1.9439605061377749E-5, 9.28892435940844E-6, 
            -4.71364248230439E-6, 4.0076997669369062E-5, -3.5484958804017966E-5, 
            -1.2043370829531207E-5, 3.0268591549819161E-5, 1.1731594766824978E-5, 
            5.6755580033076719E-5, 7.090996498299612E-5, -5.5980429844779368E-5, 
            -7.4069504977988473E-5, 1.1380044515583744E-5, -1.2478821596586743E-6, 
            -8.8556771855537856E-5, 9.9739734262247753E-5, -4.9191563892576237E-6, 
            1.6532525253799473E-5, 1.3985534576627902E-5, 2.0870603819648888E-5, 
            2.651103789024284E-6, -1.6285608007628671E-5, 9.432490086149765E-5, 
            -5.585517522439738E-6, 8.2232143628111176E-6, -1.3175532885754417E-5, 
            -1.1922328868394645E-5, -1.6623852676614686E-5, -3.2932901624606466E-5, 
            7.1469114313965659E-5, -0.00010269837772498445, 3.2001750422198327E-5, 
            -3.5610073916940805E-5, 3.5784638008169172E-5, 0.15000020984160262, 
            0.18119985040622649, 4.10045036965661E-5, 1.0010001951654295, 0.1000000075216279, 
            5.4017207288595819E-6, 0.043400002799370965]
        '''
        wTensor = torch.tensor(w, dtype=torch.float64)
        
        for layer in self.layers:
            if isinstance(layer, TanhLayer) or isinstance(layer, ReLULayer):
                
                
                layer.w.data = torch.randn_like(layer.w) * np.sqrt(2 / (layer.w.size(0) + layer.w.size(1)))
                #layer.w.data = torch.randn_like(layer.w) * 0.02 - 0.01
                layer.b.data = torch.zeros_like(layer.b)
                '''
                if wTensor.shape == layer.w.shape:
                    layer.w.data = wTensor
                    wTensor = wTensor[torch.numel(layer.w):]
                    layer.b.data = torch.zeros_like(layer.b)
                '''

            w.extend(layer.w.flatten().detach().numpy())
            w.extend(layer.b.flatten().detach().numpy())
    
        w = np.array(w)

        print("weights", w)

        return w, self

    def backpropagate(self, x):
        activations = [x]

        for layer in self.layers:

            x = x.to(dtype=torch.float64)
            x = layer(x)

            activations.append(x)
            
        # y = output
        y = activations[-1]
        tensorList = []
        DrannDw = []
        output_size = self.layers[-1].w.shape[0]
        DrannDanninp = torch.eye(output_size, dtype=torch.float64)
        A1 = DrannDanninp
        tensor_size = 0

        for i in reversed(range(len(self.layers))):
            h1 = activations[i]
            h1l = self.layers[i].derivative(h1)

            h1l_reshaped = h1l.t()
            
            h1_reshaped = torch.cat((h1.t(), torch.tensor([[1]])), dim=1)
            
            layer_dydw = torch.kron(h1_reshaped,A1)

            tensor_size = tensor_size + layer_dydw.shape[1] 
            tensorList.append(layer_dydw)

            if i == 0:
                break
            
            A1 = -(torch.mm(DrannDanninp,self.layers[i].w) * h1l_reshaped.repeat(output_size, 1))


            DrannDanninp = A1

            h1l_reshaped = torch.cat((h1l_reshaped, torch.tensor([[1]])), dim=1)

        DrannDanninp = torch.mm(A1,self.layers[0].w)


        DrannDw = tensorList

        DrannDw = torch.cat(DrannDw, dim=1)
        DrannDw = DrannDw.view(7, tensor_size)

        return y, DrannDanninp, DrannDw

class TanhLayer(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        super(TanhLayer, self).__init__()
        self.w = nn.Parameter(torch.randn(output_size, input_size, dtype=torch.float64)) 
        self.b = nn.Parameter(torch.randn(output_size, 1, dtype=torch.float64))

    def forward(self, x):

        return torch.tanh(torch.mm(self.w, x) + self.b)

    def derivative(self, x):
        
        return 1 - x ** 2


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
        self.w = nn.Parameter(torch.randn(output_size, input_size, dtype=torch.float64)) 
        self.b = nn.Parameter(torch.randn(output_size, 1, dtype=torch.float64))

    def forward(self, x):
        return torch.mm(self.w, x) + self.b

    def derivative(self, x):
        return torch.ones_like(x) 