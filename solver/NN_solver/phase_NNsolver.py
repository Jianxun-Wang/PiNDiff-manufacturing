import torch
from torch import nn

class lblock(nn.Module):
    def __init__(self, hidden_size,):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.ln = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        return self.ln(self.net(x)) + x


def gen_net(layers,insize,outsize,hsize):
    l = [nn.Linear(insize,hsize),] 
    for _ in range(layers):
        l.append(nn.LeakyReLU())
        l.append(lblock(hsize))
    l.append(nn.LeakyReLU())
    l.append(nn.Linear(hsize, outsize))
    return nn.Sequential(*l)


class Phase_NNsolver(nn.Module):   
    def __init__(self, args):
        super(Phase_NNsolver, self).__init__()
        """
        Neural ODE for solving degree of cure
        """
        # self.args = args
        self.NNDOC_layers = args.NNDOC_layers
        self.hsize=args.NNDOC_hsize

        # self.linear_layers = gen_net(   layers=args.NNDOC_layers,
        #                                 insize=2,
        #                                 outsize=1,
        #                                 hsize=args.NNDOC_hsize)

        self.inl = nn.Sequential(nn.Linear(2,self.hsize),nn.LeakyReLU())
        for i in range(self.NNDOC_layers):
            setattr(self,'l'+str(i),nn.Sequential(lblock(self.hsize),nn.LeakyReLU()))
        self.outl = nn.Sequential(nn.Linear(self.hsize,1))
            
    # Defining the forward pass    
    def forward(self, DOC_nn,theta_nn, pvalue=0.0):

        drop = nn.Dropout(p=pvalue)

        # x = torch.cat([torch.unsqueeze(DOC_nn,3),torch.unsqueeze(theta_nn,3)], dim=3)
        # DOCdot_nn = self.linear_layers(x)
        # DOCdot_nn = torch.squeeze(DOCdot_nn, 3)
        # DOCdot_nn = torch.pow(DOCdot_nn,2)

        x = torch.cat([torch.unsqueeze(DOC_nn,3),torch.unsqueeze(theta_nn,3)], dim=3)
        x = self.inl(x)
        for i in range(self.NNDOC_layers):
            x = getattr(self,'l'+str(i))(x)
            x = drop(x)
        x = self.outl(x)

        DOCdot_nn = torch.squeeze(x, 3)
        DOCdot_nn = torch.pow(DOCdot_nn,2)

        return DOCdot_nn


class Phase_INsolver(nn.Module):   
    def __init__(self, args):
        super(Phase_INsolver, self).__init__()
        """
        Inverse solver for solving degree of cure
        """
        self.Inverse_para = nn.Parameter( torch.rand((3,)) )

    # Defining the forward pass    
    def forward(self, DOC_nn,theta_nn, pvalue=0.0):

        powv = 1
        DOCdot_nn = self.Inverse_para[0]*torch.pow(DOC_nn,powv) + self.Inverse_para[1]**torch.pow(theta_nn,powv) + self.Inverse_para[2]
        DOCdot_nn = torch.pow(DOCdot_nn,2)

        return DOCdot_nn