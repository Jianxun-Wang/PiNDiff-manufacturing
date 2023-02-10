"""
Conv-LSTM Encoder-Decoder Networks

Input --> Conv --> DownSampling --> DenseBlock --> Downsampling --------
                                                                        |
Output <-- Upsampling <-- DenseBlock <-- Upsampling <-- DenseBlock <----

"""
import os
import torch
from torch._C import Size
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules import conv


class ConvLSTM(nn.Module):
    """ConvLSTMCell
    """
    def __init__(self,  input_channels, hidden_dim, kernel_size, shape, device):
        super(ConvLSTM, self).__init__()

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.shape = shape  # H, W
        # in this way the output has the same size
        self.padding = (kernel_size - 1) // 2
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_channels + self.hidden_dim,
                      4 * self.hidden_dim, self.kernel_size, 1,
                      self.padding),
            nn.GroupNorm(4 * self.hidden_dim // 32, 4 * self.hidden_dim))

    def forward(self, inputs=None, hidden_state=None, seq_len=5):
        #  seq_len=10 for moving_mnist
        if hidden_state is None:
            hx = torch.zeros(inputs.size(1), self.hidden_dim, self.shape[0],self.shape[1]).to(self.device)
            cx = torch.zeros(inputs.size(1), self.hidden_dim, self.shape[0],self.shape[1]).to(self.device)
        else:
            hx, cx = hidden_state

        if inputs is None:
            inputs = torch.zeros(seq_len, hx.size(0), self.input_channels, self.shape[0],self.shape[1]).to(self.device)


        # output_inner = []
        dims = list(inputs.size())
        dims[2] = self.hidden_dim
        # xt = torch.zeros(dims).type(inputs.type()).to(self.device)
        xt = torch.zeros(tuple(dims), dtype=inputs.dtype, device=self.device)
        for index in range(seq_len):
            x = inputs[index, ...]  #inputs[i,:,:///]

            combined = torch.cat((x, hx), 1)
            gates = self.conv(combined)  # gates: S, hidden_dim*4, H, W
            # it should return 4 tensors: i,f,g,o
            ingate, forgetgate, cellgate, outgate = torch.split(
                gates, self.hidden_dim, dim=1)
            ingate = torch.sigmoid(ingate)
            forgetgate = torch.sigmoid(forgetgate)
            cellgate = torch.tanh(cellgate)
            outgate = torch.sigmoid(outgate)

            cy = (forgetgate * cx) + (ingate * cellgate)  #Cx is the state from prev calc
            hy = outgate * torch.tanh(cy)
            # output_inner.append(hy) #stacks five input images
            xt[index, ...] = hy
            hx = hy
            cx = cy
           # print(len(output_inner))
        return xt, (hy, cy)


class ConvLSTM_Net(nn.Module):
    def __init__(self, args, nu_inp_t, in_channels, out_channels, LSTM_hc= [32,64,64]):
        super(ConvLSTM_Net, self).__init__()
        """Conv LSTM network
        """

        self.num_layers = len(LSTM_hc)
        self.nu_t = nu_inp_t
        self.LSTM_hc = LSTM_hc
        device = args.device
        num_features = self.LSTM_hc[0] // 2

        self.In_Conv = nn.Sequential()

        # First convolution, half image size ================
        self.In_Conv.add_module('In_conv3', nn.Conv2d(in_channels, num_features, 
                              kernel_size=3, stride=1, padding=1, 
                              bias=False, padding_mode='zeros'))
        self.In_Conv.add_module('In_conv_LErelu1', nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        # Encoder
        shape = [torch.tensor([args.nx, args.ny])]
        for i in range(self.num_layers-1):
            shape.append(torch.div(shape[-1],2, rounding_mode='floor'))
        # shape = torch.tensor([args.nx, args.ny])

        # convLSTM
        setattr(self, 'CLSTMEn'+str(0),
                    ConvLSTM(num_features, self.LSTM_hc[0], 
                                kernel_size = 5, shape=shape[0], device=device))
        num_features = self.LSTM_hc[0]
        
        for i in range(self.num_layers-1):
            
            setattr(self, 'Eblock'+str(i),
            nn.Sequential(nn.Conv2d(num_features, self.LSTM_hc[i], 
                            kernel_size=3, stride=2, padding=1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            num_features = self.LSTM_hc[i]

            # convLSTM
            # shape = shape//2
            setattr(self, 'CLSTMEn'+str(i+1),
                        ConvLSTM(num_features, self.LSTM_hc[i+1], 
                                 kernel_size = 5, shape=shape[i+1], device=device))
            num_features = self.LSTM_hc[i+1]

        # Decoder
        for i in range(self.num_layers-1):
            # convLSTM
            setattr(self, 'CLSTMDc'+str(i),
                        ConvLSTM(num_features, self.LSTM_hc[self.num_layers-i-1], 
                                 kernel_size = 5, shape=shape[self.num_layers-i-1], device=device))
            num_features = self.LSTM_hc[self.num_layers-i-1]
            # shape = shape*2
            
            setattr(self, 'Dblock'+str(i),
            nn.Sequential(nn.ConvTranspose2d(num_features, self.LSTM_hc[self.num_layers-i-2], 
                            kernel_size=4, stride=2, padding=1),
                        nn.LeakyReLU(negative_slope=0.2, inplace=True)))
            num_features = self.LSTM_hc[self.num_layers-i-2]
        
        # convLSTM
        setattr(self, 'CLSTMDc'+str(i+1),
                    ConvLSTM(num_features, self.LSTM_hc[0], 
                                kernel_size = 5, shape=shape[0], device=device))
        num_features = self.LSTM_hc[0]

        self.last_trans_up = nn.Sequential(nn.Conv2d(num_features, 16, 
                                kernel_size=3, stride=1,padding=1),
                             nn.Conv2d(16, out_channels, 
                                kernel_size=1, stride=1,padding=0))
                            

    def forward(self, x, EnLSTMstates_in):
        # x = x.transpose(0, 1)  # B,S,V,64,64 to S,B,V,64,64
        seq_number, batch_size, input_channel, height, width = x.size()

        x = torch.reshape(x, (-1, input_channel, height, width)) # S*B,V,64,64
        x = self.In_Conv(x)
        x = torch.reshape(x, (seq_number, batch_size, x.size(1), x.size(2), x.size(3)))  # S,B,V,64,64
        
        # Encoder
        # ConvLSTM
        x, cs = getattr(self,'CLSTMEn'+str(0))(x, EnLSTMstates_in[0], seq_len=self.nu_t)
        EnLSTMstates_out = [cs]

        for i in range(self.num_layers-1):
            # Dense Block
            x = torch.reshape(x, (-1, x.size(2), x.size(3), x.size(4))) # S*B,V,64,64
            x = getattr(self,'Eblock'+str(i))(x)
            x = torch.reshape(x, (seq_number, batch_size, x.size(1), x.size(2), x.size(3)))  # S,B,V,64,64
            # ConvLSTM
            x, cs = getattr(self,'CLSTMEn'+str(i+1))(x, EnLSTMstates_in[i+1], seq_len=self.nu_t)
            EnLSTMstates_out.append(cs) 

        # Decoder
        x = None
        for i in range(self.num_layers-1):
            # ConvLSTM
            x, _ = getattr(self,'CLSTMDc'+str(i))(x, EnLSTMstates_out[(self.num_layers-1)-i], seq_len = self.nu_t)
            # Dense Block
            x = torch.reshape(x, (-1, x.size(2), x.size(3), x.size(4))) # S*B,V,64,64
            x = getattr(self,'Dblock'+str(i))(x)
            x = torch.reshape(x, (seq_number, batch_size, x.size(1), x.size(2), x.size(3)))  # S,B,V,64,64
        
        # ConvLSTM
        x, _ = getattr(self,'CLSTMDc'+str(i+1))(x, EnLSTMstates_out[0], seq_len = self.nu_t)

        x = torch.reshape(x, (-1, x.size(2), x.size(3), x.size(4))) # S*B,V,64,64
        x = self.last_trans_up(x)
        x = torch.reshape(x, (seq_number, batch_size, x.size(1), x.size(2), x.size(3)))  # S,B,V,64,64

        # x = x.transpose(0, 1)  # S,B,V,64,64 to B,S,V,64,64
        
        return x, tuple(EnLSTMstates_out)


if __name__ == '__main__':

    device='cuda'
    ConvLSTM_ = ConvLSTM_Net(nu_inp_t = 5, in_channels = 2, out_channels = 2, LSTM_hc= [32,64,64], device=device).to(device)
    print(ConvLSTM_)
    x = 1.0*torch.zeros((10,5, 2, 64, 64), dtype=torch.float).to(device)
    x, h=ConvLSTM_(x, (None, None, None))
    print(x.size())
