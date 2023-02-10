import torch


class Phase_Solver():

    def __init__(self) -> None:
        
        self.flw_A   = torch.tensor([1.48e7, 8.30e4, 6.39e7, 9.8e4]).to(self.args.device)
        self.flw_E_R = torch.tensor([1.02e4, 8.54e3, 8.94e3, 7.1e3]).to(self.args.device)
        self.flw_m   = torch.tensor([0.17, 0.70, 1.65, 1.66]).to(self.args.device)
        self.flw_n   = torch.tensor([19.3, 0.87, 16.6, 3.9]).to(self.args.device)
        self.flw_D   = torch.tensor([0, 97.4, 0, 63.3]).to(self.args.device)
        self.flw_ac0 = torch.tensor([0, -1.6, 0, -0.6]).to(self.args.device)
        self.flw_acT = torch.tensor([0, 5.7e-3, 0, 3e-3]).to(self.args.device)

        if self.args.addNNDOC:
            if self.args.timeStepOrder == 1:
                self.step_DOC_in_time = getattr(self, 'Euiler_step_DOC_NN')
            elif self.args.timeStepOrder == 4:
                self.step_DOC_in_time = getattr(self, 'RK4_step_DOC_NN')
            else:
                raise('Set timeStepOrder = 1 0r 4')
        else:
            if self.args.timeStepOrder == 1:
                self.step_DOC_in_time = getattr(self, 'Euiler_step_DOC')
            elif self.args.timeStepOrder == 4:
                self.step_DOC_in_time = getattr(self, 'RK4_step_DOC')
            else:
                raise('Set timeStepOrder = 1 0r 4')


    def get_DOCdot_FD(self, DOC, theta):
        
        K_ = torch.zeros((4, self.args.ncases, *self.grid_shape())).to(self.args.device)
        for i in [0,1,2,3]:
            K_[i] = self.flw_A[i]*torch.exp(-self.flw_E_R[i]/theta)

        DOCdot = torch.zeros((self.args.ncases, *self.grid_shape())).to(self.args.device)
        for i in [0,2]:
            DOCdot = DOCdot + K_[i]*torch.pow(DOC,self.flw_m[i])*torch.pow((1-DOC),self.flw_n[i]) #*torch.heaviside(1-DOC,0)

        for i in [1,3]:
            DOCdot = DOCdot + K_[i]*torch.pow(DOC,self.flw_m[i])*torch.pow((1-DOC),self.flw_n[i]) \
                            / (1+torch.exp(self.flw_D[i]*(DOC-(self.flw_ac0[i]+self.flw_acT[i]*theta))))  #*torch.heaviside(1-DOC,0)
            
        return DOCdot

    def Euiler_step_DOC(self, DOC,theta):
        DOCdot = self.get_DOCdot_FD(DOC,theta)
        DOC = DOC + self.args.dt*DOCdot
        return DOC, DOCdot

    def Euiler_step_DOC_NN(self, DOC,theta):
        DOC_NN = self.get_DOCdot_NN((DOC-self.d_DOC_mean)/self.d_DOC_std,
                                    (theta-self.d_theta_mean)/self.d_theta_std,
                                    pvalue=0.0)
        DOCdot = self.d_DOC_std/(10*self.d_time_std) *DOC_NN
        DOC    = DOC + self.args.dt* DOCdot
        return DOC, DOCdot

    def RK4_step_DOC(self, DOC,theta):

        h  = self.args.dt/60.
        k1 = self.get_DOCdot_FD(DOC,theta)
        k2 = self.get_DOCdot_FD(DOC+h*k1*0.5,theta)
        k3 = self.get_DOCdot_FD(DOC+h*k2*0.5,theta)
        k4 = self.get_DOCdot_FD(DOC+h*k2,theta)
        DOCdot = 1/6 *(k1 +2*k2 +2*k3 +k4)
        DOC = DOC + self.args.dt*DOCdot
        return DOC, DOCdot

    def stepintime_DOC(self, DOC,theta):
        DOC,DOCdot = self.step_DOC_in_time(DOC,theta)
        return DOC, DOCdot
