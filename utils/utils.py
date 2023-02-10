import sys
import torch
import numpy
from utils.operators import d2udx2_2D, d2udy2_2D, dudx_2D, dudy_2D
from types import SimpleNamespace
import yaml

import argparse
import os, errno, json
from typing import List

HOME = os.getcwd()


def printProgressBar(i,max,postText):
    n_bar =20 #size of progress bar
    j= i/max
    sys.stdout.write('\r')
    sys.stdout.write(f"{postText}[{'=' * int(n_bar * j):{n_bar}s}] {int(100 * j)}% ")
    sys.stdout.flush()


class EmbeddingParser(argparse.ArgumentParser):
    """Arguments for training embedding models
    """
    def __init__(self):
        super().__init__(description='Arguments for training the embedding models for transformers of physical systems')
        tol = 1e-6

        self.add_argument('--case_name', type=str, default='0', help='Name of the case')

        # solver
        self.add_argument('--nmins', type=float, default=10, help='no. of minutes to run simulation')
        self.add_argument('--print_dtmin', type=float, default=10, help='print time step in min')
        self.add_argument('--save_dtmin', type=float, default=10, help='print time step in min')
        self.add_argument('--timeStepOrder', type=float, default=1, help='time Step Order (options: Euler = 1, RK4 = 4)')
        
        self.add_argument('--train_model', action='store_true', help='train the model using traning_data ')
        self.add_argument('--gen_syndata', action='store_true', help='generate synthetic data for traning')
        self.add_argument('--gen_rnd_batch', action='store_true', help='generate synthetic data randomly')
        self.add_argument('--train_paranorm', action='store_true', help='optimise the parameters')

        # Neural network parameters
        self.add_argument('--epochstart', type=int, default=0, help='start at epoch for traning')
        self.add_argument('--nepoch', type=int, default=1000, help='number of epochs for traning')
        self.add_argument('--lr', type=float, default=1e-2, help='learning rate for traning')
        self.add_argument('--addNNDOC', action='store_true', help='add the neural network to DOC solver')
        self.add_argument('--addNNEng', action='store_true', help='add the neural network to Energy solver')
        self.add_argument('--train_CLSTM', action='store_true', help='train using CLSMT network')
        self.add_argument('--loss_flg', type=str, help='add the neural network to Energy solver')
        self.add_argument('--NNDOC_layers', type=int, default=2, help='no of layers in DOC neural network')
        self.add_argument('--NNDOC_hsize', type=int, default=128, help='hidden layer size in DOC neural network')
        self.add_argument('--NNEng_layers', type=int, default=2, help='no of layers in N-PDE Temperature neural network')
        self.add_argument('--NNEng_hsize', type=int, default=64, help='no of channels in N-PDE Temperature neural network')
        self.add_argument('--CLSTM_hsize', type=int, default=[32,16], help='no of channels in hidden layers of ED-layers of CLSTM')
        
        self.add_argument('--Ru', type=float, default=287.058, help='Universal gas constant [J/Kg/K]')

        # # Properties
        self.add_argument('--thetastar5', type=float, default=1800.+273.15, help='temperature of solidification of meited Si [degree]')
        self.add_argument('--rhoi', type=float, default=torch.tensor([1.9, 1.1, 1.7, 1.2e-3, 1.1, 2.3, 2.3, 3.2, 2.1])*1e3, help='density of each phase [kg/m^3]')
        self.add_argument('--J0', type=float, default=torch.tensor([10, 3e5, 10, 0, 100, 100, 0., 0., 0.]), help='physico-chemical transformation: constant [kgs/m^3]')
        self.add_argument('--Ea', type=float, default=torch.tensor([3., 3.2, 3.5, 0., 1., 4., 0., 0., 0.])*1e6, help='physico-chemical transformation: Activation energy [J/kg]')
        self.add_argument('--Cspht', type=float, default=torch.tensor([0.9, 1.1, 1.1, 1.01, 1., 1., 1., 1., 1.])*1e3, help='specific heat capicity [J/Kg/K]')
        self.add_argument('--Hphase', type=float, default=torch.tensor([0., 0., 0., 0., -575.e3, 0., 0., 0., 0.]), help='Heat of phase transformation []')
        
        # self.add_argument('--plot_dir', type=str, default="plots", help='Universal gas constant [J/Kg/K]')

    def mkdirs(self, *directories: str) -> None:
        """Makes a directory if it does not exist

        Args:
           directories (str...): a sequence of directories to create

        Raises:
            OSError: if directory cannot be created
        """
        for directory in list(directories):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def parse(self, args:List = None, dirs: bool = True) -> None:
        """Parse program arguments

        Args:
            args (List, optional): Explicit list of arguments. Defaults to None.
            dirs (bool, optional): Make experiment directories. Defaults to True.
        """
        if args:
            args = self.parse_args(args=args)
        else:
            args = self.parse_args()

        args.in_dir   = os.path.join(HOME, "input")
        args.out_dir  = os.path.join(HOME, "output"+"/output_"+args.case_name)
        args.plot_dir = os.path.join(HOME, "output"+"/output_"+args.case_name, "plots")
        args.pred_dir = os.path.join(HOME, "output"+"/output_"+args.case_name, "plots", "prediction")
        args.val_dir  = os.path.join(HOME, "output"+"/output_"+args.case_name, "plots", "validation")
        args.ckpt_dir = os.path.join(HOME, "output"+"/output_"+args.case_name, "checkpoints")

        self.mkdirs(args.in_dir)
        self.mkdirs(args.out_dir)
        self.mkdirs(args.plot_dir)
        self.mkdirs(args.pred_dir)
        self.mkdirs(args.val_dir)
        self.mkdirs(args.ckpt_dir)

        # read parameters from case_setup.yaml file
        case_setup_params = SimpleNamespace(**yaml.load(open(args.in_dir+"/case_setup.yaml"), Loader=yaml.FullLoader))
        for member in case_setup_params.__dict__:
            args.__dict__[member] = case_setup_params.__dict__[member]

        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)

        print("device = ",args.device)
        if hasattr(case_setup_params, 'set_num_threads'):
            torch.set_num_threads(case_setup_params.set_num_threads)

        return args





# def Grad2D(Var, grid2D, VarBC=[], device = 'cpu'):
    
#     dudx = dudx_2D(scheme='Central1', accuracy = 2)
#     dudy = dudy_2D(scheme='Central1', accuracy = 2)

#     if VarBC: 
#         [VarLeft, VarRight, VarBottom, VarTop] = VarBC
#         Grad_x = torch.zeros((1,1,*grid2D.grid_shape())).to(device)
#         Grad_y = torch.zeros((1,1,*grid2D.grid_shape())).to(device)
#         VarBC = Var.clone()
#         VarBC[ 0,:] = VarLeft
#         VarBC[-1,:] = VarRight
#         VarBC[:, 0] = VarBottom
#         VarBC[:,-1] = VarTop
#         Grad_x[:,:,1:-1,1:-1] = (dudx(VarBC[None,None,:,1:-1]))/grid2D.grid_dx
#         Grad_y[:,:,1:-1,1:-1] = (dudy(VarBC[None,None,1:-1,:]))/grid2D.grid_dy
#     else: # Periodic
#         VarBC = torch.zeros((grid2D.grid_shape()[0]+2, grid2D.grid_shape()[1]+2)).to(device)
#         VarBC[1:-1,1:-1] = Var.clone()
#         VarBC[ 0,1:-1] = Var[-1,:]
#         VarBC[-1,1:-1] = Var[ 0,:]
#         VarBC[1:-1, 0] = Var[:,-1]
#         VarBC[1:-1,-1] = Var[:, 0]
#         Grad_x = (dudx(VarBC[None,None,:,1:-1]))/grid2D.grid_dx
#         Grad_y = (dudy(VarBC[None,None,1:-1,:]))/grid2D.grid_dy

#     return Grad_x[0,0], Grad_y[0,0]

# def Div2D(Varx, Vary, grid2D, VarBC=[], device = 'cpu'):
    
#     dudx = dudx_2D(scheme='Central1', accuracy = 2)
#     dudy = dudy_2D(scheme='Central1', accuracy = 2)

#     if VarBC: 
#         [VarLeft, VarRight, VarBottom, VarTop] = VarBC
#         Div = torch.zeros((1,1,*grid2D.grid_shape())).to(device)
#         VarxBC = Varx.clone()
#         VaryBC = Vary.clone()
#         VarxBC[ 0,:] = VarLeft
#         VarxBC[-1,:] = VarRight
#         VaryBC[:, 0] = VarBottom
#         VaryBC[:,-1] = VarTop
#         Div[:,:,1:-1,1:-1] = (dudx(VarxBC[None,None,:,1:-1])) /grid2D.grid_dx  \
#                             +(dudy(VaryBC[None,None,1:-1,:])) /grid2D.grid_dy
#     else: # Periodic
#         VarxBC = torch.zeros((grid2D.grid_shape()[0]+2, grid2D.grid_shape()[1]+2)).to(device)
#         VaryBC = torch.zeros((grid2D.grid_shape()[0]+2, grid2D.grid_shape()[1]+2)).to(device)
#         VarxBC[1:-1,1:-1] = Varx.clone()
#         VaryBC[1:-1,1:-1] = Vary.clone()
#         VarxBC[ 0,1:-1] = Varx[-1,:]
#         VarxBC[-1,1:-1] = Varx[ 0,:]
#         VaryBC[1:-1, 0] = Vary[:,-1]
#         VaryBC[1:-1,-1] = Vary[:, 0]
#         Div             = (dudx(VarxBC[None,None,:,1:-1])) /grid2D.grid_dx  \
#                          +(dudy(VaryBC[None,None,1:-1,:])) /grid2D.grid_dy

#     return Div[0,0]

# def Laplace2D(Var, grid2D, VarBC=[],k=[1,1], device = 'cpu'):
    
#     kx, ky = k
    
#     d2udx2 = d2udx2_2D(scheme='Central2', accuracy = 2)
#     d2udy2 = d2udy2_2D(scheme='Central2', accuracy = 2)

#     if VarBC: 
#         [VarLeft, VarRight, VarBottom, VarTop] = VarBC
#         Lap = torch.zeros((1,1,*grid2D.grid_shape())).to(device)
#         VarBC = Var.clone()
#         VarBC[ 0,:] = VarLeft
#         VarBC[-1,:] = VarRight
#         VarBC[:, 0] = VarBottom
#         VarBC[:,-1] = VarTop
#         Lap[:,:,1:-1,1:-1]  = kx*(d2udx2(VarBC[None,None,:,1:-1]))/(grid2D.grid_dx*grid2D.grid_dx)  \
#                              +ky*(d2udy2(VarBC[None,None,1:-1,:]))/(grid2D.grid_dy*grid2D.grid_dy)
#     else: # Periodic
#         VarBC = torch.zeros((grid2D.grid_shape()[0]+2, grid2D.grid_shape()[1]+2)).to(device)
#         VarBC[1:-1,1:-1] = Var.clone()
#         VarBC[ 0,1:-1] = Var[-1,:]
#         VarBC[-1,1:-1] = Var[ 0,:]
#         VarBC[1:-1, 0] = Var[:,-1]
#         VarBC[1:-1,-1] = Var[:, 0]
#         Lap                = kx*(d2udx2(VarBC[None,None,:,1:-1]))/(grid2D.grid_dx*grid2D.grid_dx)  \
#                             +ky*(d2udy2(VarBC[None,None,1:-1,:]))/(grid2D.grid_dy*grid2D.grid_dy)

#     return Lap[0,0]


#####################################################################################################


def Grad2D(Var, grid2D, VarBC=[], device = 'cpu'):
    
    if VarBC: 
        [VarLeft, VarRight, VarBottom, VarTop] = VarBC
        Grad_x = torch.zeros(grid2D.grid_shape()).to(device)
        Grad_y = torch.zeros(grid2D.grid_shape()).to(device)
        VarBC = Var.clone()
        VarBC[ 0,:] = VarLeft
        VarBC[-1,:] = VarRight
        VarBC[:, 0] = VarBottom
        VarBC[:,-1] = VarTop
        Grad_x[1:-1,1:-1] = (VarBC[2:,1:-1] - VarBC[:-2,1:-1])/(2*grid2D.grid_dx)
        Grad_y[1:-1,1:-1] = (VarBC[1:-1,2:] - VarBC[1:-1,:-2])/(2*grid2D.grid_dy)
    else: # Periodic
        VarBC = torch.zeros((grid2D.grid_shape()[0]+2, grid2D.grid_shape()[1]+2)).to(device)
        VarBC[1:-1,1:-1] = Var.clone()
        VarBC[ 0,1:-1] = Var[-1,:]
        VarBC[-1,1:-1] = Var[ 0,:]
        VarBC[1:-1, 0] = Var[:,-1]
        VarBC[1:-1,-1] = Var[:, 0]
        Grad_x = (VarBC[2:,1:-1] - VarBC[:-2,1:-1])/(2*grid2D.grid_dx)
        Grad_y = (VarBC[1:-1,2:] - VarBC[1:-1,:-2])/(2*grid2D.grid_dy)

    return Grad_x, Grad_y

def Div2D(Varx, Vary, grid2D, VarBC=[], device = 'cpu'):
    
    if VarBC: 
        [VarLeft, VarRight, VarBottom, VarTop] = VarBC
        Div = torch.zeros(grid2D.grid_shape()).to(device)
        VarxBC = Varx.clone()
        VaryBC = Vary.clone()
        VarxBC[ 0,:] = VarLeft
        VarxBC[-1,:] = VarRight
        VaryBC[:, 0] = VarBottom
        VaryBC[:,-1] = VarTop
        Div[1:-1,1:-1] = (VarxBC[2:,1:-1] - VarxBC[:-2,1:-1])/(2*grid2D.grid_dx)  \
                        +(VaryBC[1:-1,2:] - VaryBC[1:-1,:-2])/(2*grid2D.grid_dy)

        # Div[1:-1,1:-1]  = (VarxBC[2:,1:-1] - VarxBC[1:-1,1:-1])/grid2D.grid_dx *(VarxBC[1:-1,1:-1]<=0)\
        #                 + (VarxBC[1:-1,1:-1] - VarxBC[:-2,1:-1])/grid2D.grid_dx *(VarxBC[1:-1,1:-1]>0)\
        #                 + (VaryBC[1:-1,2:] - VaryBC[1:-1,1:-1])/grid2D.grid_dy *(VaryBC[1:-1,1:-1]<=0)\
        #                 + (VaryBC[1:-1,1:-1] - VaryBC[1:-1,:-2])/grid2D.grid_dy *(VaryBC[1:-1,1:-1]>0)\

    else: # Periodic
        VarxBC = torch.zeros((grid2D.grid_shape()[0]+2, grid2D.grid_shape()[1]+2)).to(device)
        VaryBC = torch.zeros((grid2D.grid_shape()[0]+2, grid2D.grid_shape()[1]+2)).to(device)
        VarxBC[1:-1,1:-1] = Varx.clone()
        VaryBC[1:-1,1:-1] = Vary.clone()
        VarxBC[ 0,1:-1] = Varx[-1,:]
        VarxBC[-1,1:-1] = Varx[ 0,:]
        VaryBC[1:-1, 0] = Vary[:,-1]
        VaryBC[1:-1,-1] = Vary[:, 0]
        Div = (VarxBC[2:,1:-1] - VarxBC[:-2,1:-1])/(2*grid2D.grid_dx) \
            + (VaryBC[1:-1,2:] - VaryBC[1:-1,:-2])/(2*grid2D.grid_dy)
        
        # Div = (VarxBC[2:,1:-1] - VarxBC[1:-1,1:-1])/grid2D.grid_dx *(VarxBC[1:-1,1:-1]<=0)\
        #     + (VarxBC[1:-1,1:-1] - VarxBC[:-2,1:-1])/grid2D.grid_dx *(VarxBC[1:-1,1:-1]>0)\
        #     + (VaryBC[1:-1,2:] - VaryBC[1:-1,1:-1])/grid2D.grid_dy *(VaryBC[1:-1,1:-1]<=0)\
        #     + (VaryBC[1:-1,1:-1] - VaryBC[1:-1,:-2])/grid2D.grid_dy *(VaryBC[1:-1,1:-1]>0)\

    return Div

def Laplace2D(Var, grid2D, VarBC=[], k = [1,1], device = 'cpu'):
    
    kx, ky = k

    if VarBC: 
        [VarLeft, VarRight, VarBottom, VarTop] = VarBC
        Lap = torch.zeros((len(Var), *grid2D.grid_shape())).to(device)
        VarBC = Var.clone()
        VarBC[:, 0,:] = VarLeft
        VarBC[:,-1,:] = VarRight
        VarBC[:,:, 0] = VarBottom
        VarBC[:,:,-1] = VarTop
        Lap[:,1:-1,1:-1] = kx*(VarBC[:,2:,1:-1] + VarBC[:,:-2,1:-1] - 2*VarBC[:,1:-1,1:-1])/(grid2D.grid_dx*grid2D.grid_dx)  \
                          +ky*(VarBC[:,1:-1,2:] + VarBC[:,1:-1,:-2] - 2*VarBC[:,1:-1,1:-1])/(grid2D.grid_dy*grid2D.grid_dy)
    else: # Periodic
        VarBC = torch.zeros((len(Var), grid2D.grid_shape()[0]+2, grid2D.grid_shape()[1]+2)).to(device)
        VarBC[:,1:-1,1:-1] = Var.clone()
        VarBC[:, 0,1:-1] = Var[:,-1,:]
        VarBC[:,-1,1:-1] = Var[:, 0,:]
        VarBC[:,1:-1, 0] = Var[:,:,-1]
        VarBC[:,1:-1,-1] = Var[:,:, 0]
        Lap            = kx*(VarBC[:,2:,1:-1] + VarBC[:,:-2,1:-1] - 2*VarBC[:,1:-1,1:-1])/(grid2D.grid_dx*grid2D.grid_dx)  \
                        +ky*(VarBC[:,1:-1,2:] + VarBC[:,1:-1,:-2] - 2*VarBC[:,1:-1,1:-1])/(grid2D.grid_dy*grid2D.grid_dy)

    return Lap

def Fahrenheit2K(theta):
    theta = (theta + 459.67)*5/9
    return theta

def K2Fahrenheit(theta):
    theta =theta*9/5 - 459.67
    return theta