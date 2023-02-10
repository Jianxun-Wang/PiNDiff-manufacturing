import numpy as np
import torch
from utils.utils import Laplace2D


class Energy_solver:

    def __init__(self) -> None:

        vr_ = self.args.volFrRes
        vf_ = self.args.volFrFib
        kr_ = self.args.kRes
        kf_ = self.args.kFib
        B_  = 2*(kr_/kf_ - 1)
        self.temp_kxx = vr_*kr_ + vf_*kf_
        self.temp_kyy = ((1-2*np.sqrt(vf_/np.pi))  \
            +1/B_*(np.pi-4/np.sqrt(1-B_*B_*vf_/np.pi))  \
            *np.arctan(np.sqrt(1-B_*B_*vf_/np.pi)/(1+B_*np.sqrt(vf_/np.pi)))) * kr_

        if self.args.addNNEng:
            if self.args.timeStepOrder == 1:
                self.step_T_in_time = getattr(self, 'Euiler_step_theta_NN')
            elif self.args.timeStepOrder == 4:
                self.step_T_in_time = getattr(self, 'RK4_step_theta_NN')
            else:
                raise('Set timeStepOrder = 1 0r 4')
        else:
            if self.args.timeStepOrder == 1:
                self.step_T_in_time = getattr(self, 'Euiler_step_theta')
            elif self.args.timeStepOrder == 4:
                self.step_T_in_time = getattr(self, 'RK4_step_theta')
            else:
                raise('Set timeStepOrder = 1 0r 4')

    
    def get_theta_dot(self, timemin,theta,DOCdot,para):
            
        Hrxn = para[1]  # self.args.Hrxn
        if len(para) > 2:
            kxx, kyy = para[2], para[3]
        else:
            kxx, kyy = self.temp_kxx, self.temp_kyy

        HeatGen = self.args.volFrRes*self.args.rhoRes*Hrxn*DOCdot
        Heatdiff = Laplace2D(theta, self, k=[kxx,kyy], device=self.args.device)
        
        # theta[timeidx+1] = T_BC*torch.ones((ccc.grid_shape()))
        if (pow(min(self.grid_dx,self.grid_dy),2)/(4*kxx)*(self.args.rhoCom*self.args.CspCom) < self.args.dt):
            print(pow(min(self.grid_dx,self.grid_dy),2)/(4*kxx)*(self.args.rhoCom*self.args.CspCom), self.args.dt)
            raise('dt issue, need to change array size')

        theta_dot = (Heatdiff + HeatGen) / (self.args.rhoCom*self.args.CspCom)

        return theta_dot

    def update_BC_theta(self, timemin,theta,DOCdot,para,BC='fixed'):

        T_out = torch.zeros_like(theta)
        for cases in range(self.args.ncases):
            T_out_ = self.getT_BC(timemin,cases)
            T_out[cases, 0,:] = T_out_
            T_out[cases,-1,:] = T_out_
            T_out[cases,:, 0] = T_out_
            T_out[cases,:,-1] = T_out_

        if BC == 'fixed':
            theta += T_out
        elif BC == 'convective':
            # h = 30#150
            h = para[0]
            Hrxn = para[1] # self.args.Hrxn
            if len(para) > 2:
                kxx, kyy = para[2], para[3]
            else:
                kxx, kyy = self.temp_kxx, self.temp_kyy

            dx, dy = self.grid_dx, self.grid_dy
            DNR = (kxx*dy/dx + kyy*dx/dy)

            # HeatGen = self.args.volFrRes*self.args.rhoRes*Hrxn*DOCdot
            # HeatGen[:, 0,1:-1] = 0.75*HeatGen[:, 0,1:-1] + 0.25*HeatGen[:, 1,1:-1]
            # HeatGen[:,-1,1:-1] = 0.75*HeatGen[:,-1,1:-1] + 0.25*HeatGen[:,-2,1:-1]
            # HeatGen[:,1:-1, 0] = 0.75*HeatGen[:,1:-1, 0] + 0.25*HeatGen[:,1:-1, 1]
            # HeatGen[:,1:-1,-1] = 0.75*HeatGen[:,1:-1,-1] + 0.25*HeatGen[:,1:-1,-2]
            # HeatGen[:, 0, 0]   = 0.5*HeatGen[:, 0, 0] + 0.25*(HeatGen[:, 1, 0]+HeatGen[:, 0, 1])
            # HeatGen[:,-1, 0]   = 0.5*HeatGen[:,-1, 0] + 0.25*(HeatGen[:,-1, 1]+HeatGen[:,-2, 0])
            # HeatGen[:, 0,-1]   = 0.5*HeatGen[:, 0,-1] + 0.25*(HeatGen[:, 1,-1]+HeatGen[:, 0,-2])
            # HeatGen[:,-1,-1]   = 0.5*HeatGen[:,-1,-1] + 0.25*(HeatGen[:,-1,-2]+HeatGen[:,-2,-1])

            HeatGen_ = self.args.volFrRes*self.args.rhoRes*Hrxn*DOCdot
            HeatGen  = HeatGen_[:,1:-1,1:-1]
            HeatGenl = 0.75*HeatGen_[:, 0:1,1:-1] + 0.25*HeatGen_[:, 1:2,1:-1]
            HeatGenr = 0.75*HeatGen_[:,-1: ,1:-1] + 0.25*HeatGen_[:,-2:-1,1:-1]
            HeatGenb = 0.75*HeatGen_[:,1:-1, 0:1] + 0.25*HeatGen_[:,1:-1, 1:2]
            HeatGent = 0.75*HeatGen_[:,1:-1,-1: ] + 0.25*HeatGen_[:,1:-1,-2:-1]
            HeatGenlb   = 0.5*HeatGen_[:, 0:1, 0:1] + 0.25*(HeatGen_[:, 1:2, 0:1]+HeatGen_[:, 0:1, 1:2])
            HeatGenrb   = 0.5*HeatGen_[:,-1: , 0:1] + 0.25*(HeatGen_[:,-1: , 1:2]+HeatGen_[:,-2:-1, 0:1])
            HeatGenlt   = 0.5*HeatGen_[:, 0:1,-1: ] + 0.25*(HeatGen_[:, 1:2,-1: ]+HeatGen_[:, 0:1,-2:-1])
            HeatGenrt   = 0.5*HeatGen_[:,-1: ,-1: ] + 0.25*(HeatGen_[:,-1: ,-2:-1]+HeatGen_[:,-2:-1,-1: ])
            HeatGenl = torch.cat((HeatGenlb,HeatGenl,HeatGenlt), dim=2)
            HeatGenr = torch.cat((HeatGenrb,HeatGenr,HeatGenrt), dim=2)
            HeatGen  = torch.cat((HeatGenb,HeatGen,HeatGent), dim=2)
            HeatGen  = torch.cat((HeatGenl,HeatGen,HeatGenr), dim=1)

            Tout_theta = T_out-theta
            thetabc = theta[:,1:-1, 1:-1]

            # theta[:, 0,1:-1] = (h*(Tout_theta[:, 0,1:-1])*dy +0.5*HeatGen[:, 0,1:-1]*dx*dy   \
            #                  +kxx*dy/dx*theta[:, 1,1:-1] +kyy*dx/dy*0.5*(theta[:, 0,2:]+theta[:, 0,:-2])) /DNR
            # theta[:,-1,1:-1] = (h*(Tout_theta[:,-1,1:-1])*dy +0.5*HeatGen[:,-1,1:-1]*dx*dy   \
            #                  +kxx*dy/dx*theta[:,-2,1:-1] +kyy*dx/dy*0.5*(theta[:,-1,2:]+theta[:,-1,:-2])) /DNR
            # theta[:,1:-1, 0] = (h*(Tout_theta[:,1:-1, 0])*dx +0.5*HeatGen[:,1:-1, 0]*dx*dy   \
            #                  +kyy*dx/dy*theta[:,1:-1, 1] +kxx*dy/dx*0.5*(theta[:,2:, 0]+theta[:,:-2, 0])) /DNR
            # theta[:,1:-1,-1] = (h*(Tout_theta[:,1:-1,-1])*dx +0.5*HeatGen[:,1:-1,-1]*dx*dy   \
            #                  +kyy*dx/dy*theta[:,1:-1,-2] +kxx*dy/dx*0.5*(theta[:,2:,-1]+theta[:,:-2,-1])) /DNR

            # theta[:, 0, 0] = (h*(Tout_theta[:, 0, 0])/(2*np.sqrt(2))*(dx+dy) +0.25*HeatGen[:, 0, 0]*dx*dy 
            #                 +0.5*kxx*dy/dx*theta[:, 1, 0] +0.5*kyy*dx/dy*theta[:, 0, 1]) /(0.5*DNR)
            # theta[:, 0,-1] = (h*(Tout_theta[:, 0,-1])/(2*np.sqrt(2))*(dx+dy) +0.25*HeatGen[:, 0,-1]*dx*dy 
            #                 +0.5*kxx*dy/dx*theta[:, 1,-1] +0.5*kyy*dx/dy*theta[:, 0,-2]) /(0.5*DNR)
            # theta[:,-1, 0] = (h*(Tout_theta[:,-1, 0])/(2*np.sqrt(2))*(dx+dy) +0.25*HeatGen[:,-1, 0]*dx*dy 
            #                 +0.5*kxx*dy/dx*theta[:,-2, 0] +0.5*kyy*dx/dy*theta[:,-1, 1]) /(0.5*DNR)
            # theta[:,-1,-1] = (h*(Tout_theta[:,-1,-1])/(2*np.sqrt(2))*(dx+dy) +0.25*HeatGen[:,-1,-1]*dx*dy 
            #                 +0.5*kxx*dy/dx*theta[:,-2,-1] +0.5*kyy*dx/dy*theta[:,-1,-2]) /(0.5*DNR)

            thetal = (h*(Tout_theta[:, 0:1,1:-1])*dy +0.5*HeatGen[:, 0:1,1:-1]*dx*dy   \
                     +kxx*dy/dx*theta[:, 1:2,1:-1] +kyy*dx/dy*0.5*(theta[:, 0:1,2:]+theta[:, 0:1,:-2])) /DNR
            thetar = (h*(Tout_theta[:,-1: ,1:-1])*dy +0.5*HeatGen[:,-1: ,1:-1]*dx*dy   \
                     +kxx*dy/dx*theta[:,-2:-1,1:-1] +kyy*dx/dy*0.5*(theta[:,-1: ,2:]+theta[:,-1: ,:-2])) /DNR
            thetab = (h*(Tout_theta[:,1:-1, 0:1])*dx +0.5*HeatGen[:,1:-1, 0:1]*dx*dy   \
                     +kyy*dx/dy*theta[:,1:-1, 1:2] +kxx*dy/dx*0.5*(theta[:,2:, 0:1]+theta[:,:-2, 0:1])) /DNR
            thetat = (h*(Tout_theta[:,1:-1,-1: ])*dx +0.5*HeatGen[:,1:-1,-1: ]*dx*dy   \
                     +kyy*dx/dy*theta[:,1:-1,-2:-1] +kxx*dy/dx*0.5*(theta[:,2:,-1: ]+theta[:,:-2,-1: ])) /DNR

            thetalb = (h*(Tout_theta[:, 0:1, 0:1])/(2*np.sqrt(2))*(dx+dy) +0.25*HeatGen[:, 0:1, 0:1]*dx*dy 
                      +0.5*kxx*dy/dx*theta[:, 1:2, 0:1] +0.5*kyy*dx/dy*theta[:, 0:1, 1:2]) /(0.5*DNR)
            thetalt = (h*(Tout_theta[:, 0:1,-1: ])/(2*np.sqrt(2))*(dx+dy) +0.25*HeatGen[:, 0:1,-1: ]*dx*dy 
                      +0.5*kxx*dy/dx*theta[:, 1:2,-1: ] +0.5*kyy*dx/dy*theta[:, 0:1,-2:-1]) /(0.5*DNR)
            thetarb = (h*(Tout_theta[:,-1: , 0:1])/(2*np.sqrt(2))*(dx+dy) +0.25*HeatGen[:,-1: , 0:1]*dx*dy 
                      +0.5*kxx*dy/dx*theta[:,-2:-1, 0:1] +0.5*kyy*dx/dy*theta[:,-1: , 1:2]) /(0.5*DNR)
            thetart = (h*(Tout_theta[:,-1: ,-1: ])/(2*np.sqrt(2))*(dx+dy) +0.25*HeatGen[:,-1: ,-1: ]*dx*dy 
                      +0.5*kxx*dy/dx*theta[:,-2:-1,-1: ] +0.5*kyy*dx/dy*theta[:,-1: ,-2:-1]) /(0.5*DNR)

            thetal  = torch.cat((thetalb,thetal,thetalt), dim=2)
            thetar  = torch.cat((thetarb,thetar,thetart), dim=2)
            thetabc = torch.cat((thetab,thetabc,thetat), dim=2)
            thetabc = torch.cat((thetal,thetabc,thetar), dim=1)
        return thetabc

    def Euiler_step_theta(self, timemin,theta,DOCdot,para,BC):
        
        theta_dot = self.get_theta_dot(timemin,theta,DOCdot,para)
        theta = theta + self.args.dt*theta_dot

        theta = self.update_BC_theta(timemin,theta,DOCdot,para,BC)
        return theta

    def Euiler_step_theta_NN(self, timemin,theta,DOCdot,para,BC):
        
        T_out = torch.zeros_like(theta).to(self.args.device)
        for cases in range(self.args.ncases):
            T_out[cases] = self.getT_BC(timemin,cases)
        
        # theta_NN = self.get_thetadot_NN((theta.detach()-self.d_theta_mean)/self.d_theta_std,
        #                                 DOCdot*1e3, para,
        #                                 (T_out.detach()-self.d_theta_mean)/self.d_theta_std)
        theta_NN = self.get_thetadot_NN((theta-self.d_theta_mean)/self.d_theta_std,
                                        (DOCdot-self.d_DOC_mean)/self.d_DOC_std, para,
                                        (T_out-self.d_theta_mean)/self.d_theta_std)
                                        
        theta_dot = self.d_theta_std/(10*self.d_time_std) *theta_NN
        theta = theta + self.args.dt*theta_dot

        theta = self.update_BC_theta(timemin,theta,DOCdot,para,BC)
        return theta

    def RK4_step_theta(self, timemin,theta,DOCdot,para,BC):

        h  = self.args.dt/3600.
        k1 = self.get_theta_dot(timemin,      theta,         DOCdot,para)
        k2 = self.get_theta_dot(timemin+h*0.5,theta+h*k1*0.5,DOCdot,para)
        k3 = self.get_theta_dot(timemin+h*0.5,theta+h*k2*0.5,DOCdot,para)
        k4 = self.get_theta_dot(timemin+h,    theta+h*k2,    DOCdot,para)
        theta = theta + self.args.dt/6 *(k1 +2*k2 +2*k3 +k4)

        theta = self.update_BC_theta(timemin,theta,DOCdot,para,BC)
        return theta

    def stepintime_theta(self, timemin,theta,DOCdot,para,BC='fixed'):
        theta = self.step_T_in_time(timemin,theta,DOCdot,para,BC)
        return theta