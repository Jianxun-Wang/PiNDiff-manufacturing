import torch
from utils.utils import Fahrenheit2K

tol = 1e-6

class grid:
    def __init__(self, nx, ny, lx=[0,1], ly=[0,1]) -> None:
        self.nx, self.ny = nx, ny
        x = torch.linspace(lx[0], lx[1], self.nx)
        y = torch.linspace(ly[0], ly[1], self.ny)
        self.x, self.y = torch.meshgrid(x, y, indexing='ij')
        self.grid_dx = (lx[1]-lx[0])/(self.nx-1)
        self.grid_dy = (ly[1]-ly[0])/(self.ny-1)

    def grid_shape(self):
        return self.nx, self.ny

    def disp(self):
        print("x=", self.x)
        print("y=", self.y)

class BC_Temperature:
    def __init__(self, timeminsBC=[], thetavalBC=[]) -> None:
        self.timeminsBC = timeminsBC
        self.thetavalBC = thetavalBC
    
    def getT_BC(self, time, case = 0):
        baseT = 0.
        if time >= self.timeminsBC[case][0] and time <= self.timeminsBC[case][-1]:
            min = self.timeminsBC[case]
            T  = self.thetavalBC[case]
            for i in range(1,len(min)):
                if time <= min[i]:
                    return Fahrenheit2K(T[i-1] + (time-min[i-1])/(min[i]-min[i-1]) * (T[i]-T[i-1]) + baseT)
        else:
            raise Exception("time should be between [",self.timeminsBC[case][0], self.timeminsBC[case][-1],"]")
