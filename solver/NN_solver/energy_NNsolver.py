import torch
from torch import nn
import torch.nn.functional as F

class cblock(nn.Module):
    def __init__(self,hc,ksize,feature_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(hc,hc,ksize,padding=ksize//2),
            nn.LeakyReLU(),
            nn.Conv2d(hc,hc,ksize,padding=ksize//2),
        )
        self.ln = nn.LayerNorm([hc]+feature_size)
        
    def forward(self,x):
        return self.ln(self.net(x)) + x


def gen_net(layers,insize,outsize,hsize,feature_size):
    l = [nn.Conv2d(insize, hsize, kernel_size=3, stride=1, padding=1),] 
    for _ in range(layers):
        l.append(nn.LeakyReLU())
        l.append(cblock(hsize,ksize=3,feature_size=feature_size))
    l.append(nn.LeakyReLU())
    l.append(nn.Conv2d(hsize, outsize, kernel_size=3, stride=1, padding=1))
    return nn.Sequential(*l)


class Energy_NNsolver(nn.Module):   
    def __init__(self, args):
        super(Energy_NNsolver, self).__init__()
        """
        Neural PDE for solving energy equation
        """
        self.args = args

        nx = self.args.nx# + 2
        ny = self.args.ny# + 2
        feature_size=[nx, ny]
        self.cnn_layers = gen_net(  layers=args.NNEng_layers,
                                    insize=3,
                                    outsize=1,
                                    hsize=args.NNEng_hsize,
                                    feature_size=feature_size)

    # Defining the forward pass    
    def forward(self,theta_nn,DOCdot_nn,para,T_out):
        
        # theta_nn  = F.pad(theta_nn, [1,1,1,1], "constant", T_out.item()).unsqueeze(1)
        # DOCdot_nn = F.pad(DOCdot_nn,[1,1,1,1], "constant", 0).unsqueeze(1)
        # theta_h   = (torch.ones_like(DOCdot_nn)*para[0])
        # x = torch.cat([DOCdot_nn, theta_nn, theta_h], dim=1)

        DOCdot_nn = torch.unsqueeze(DOCdot_nn,1)
        theta_nn = torch.unsqueeze(theta_nn,1)
        theta_BC = torch.unsqueeze(T_out,1)
        # theta_h  = torch.ones_like(theta_nn)*para[0]
        x = torch.cat([DOCdot_nn, theta_nn, theta_BC], dim=1)

        DOCdot_nn = self.cnn_layers(x)
        DOCdot_nn = torch.squeeze(DOCdot_nn, 1)

        return DOCdot_nn