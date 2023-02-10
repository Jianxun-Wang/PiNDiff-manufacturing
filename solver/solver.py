from dataclasses import replace
import numpy as np
import torch
from torch import nn, tensor
import h5py
from solver.FD_solver.grid_bc import grid, BC_Temperature
from solver.FD_solver.phase_solver import Phase_Solver
from solver.FD_solver.energy_solver import Energy_solver
from solver.NN_solver.ConvLSTM_Solver import ConvLSTM_Net
from utils.utils import K2Fahrenheit
from utils.post import plot_valdation

# from torch.profiler import profile, record_function, ProfilerActivity
# from memory_profiler import profile

class cccSolver(grid,
                BC_Temperature, 
                Phase_Solver,
                Energy_solver):

    def __init__(self, args,name="traning_data") -> None:
        self.args = args
        grid.__init__(self,self.args.nx, self.args.ny, self.args.lx, self.args.ly)
        Phase_Solver.__init__(self)
        Energy_solver.__init__(self)

        if not args.gen_syndata:
            self.Data_Exp = self.load_states(name)
            self.d_DOC    = self.Data_Exp["DOC"].detach().to(self.args.device)
            self.d_theta  = self.Data_Exp["theta"].detach().to(self.args.device)
            self.d_DOC    = torch.reshape(self.d_DOC,  (-1,self.args.ncases,self.args.nx*self.args.ny))
            self.d_theta  = torch.reshape(self.d_theta,(-1,self.args.ncases,self.args.nx*self.args.ny))
            if args.train_model:
                self.d_DOC_mean   = self.d_DOC.mean()
                self.d_DOC_std    = self.d_DOC.std()
                self.d_theta_mean = self.d_theta.mean()
                self.d_theta_std  = self.d_theta.std()
                self.d_time_std   = (self.Data_Exp["timemin"]*60.).std()
                # self.d_time_std   = self.Data_Exp["timemin"][-1]*60.
            
            if args.train_CLSTM:
                self.train = getattr(self, 'train_CLSTMNet')
                self.forward = getattr(self, 'CLSTMNet')
            else:
                self.train = getattr(self, 'train_Hybrid_solver')
                self.forward = getattr(self, 'Hybrid_solver')

        BC_Temperature.__init__(self,self.args.timeminsBC, self.args.thetavalBC)

        if self.args.target_sparsedata > 0:
            if self.args.target_sparsedata > 1.:
                len_sparsedata = self.args.target_sparsedata
            else:
                len_sparsedata = self.args.target_sparsedata * self.args.nx*self.args.ny
            self.sparse_data_idx = torch.randperm(self.args.nx*self.args.ny)
            self.sparse_data_idx = self.sparse_data_idx[:int(len_sparsedata)]
        else:
            x = torch.linspace(0, self.args.nx, -self.args.target_sparsedata+2, dtype=torch.int64)
            y = torch.linspace(0, self.args.ny, -self.args.target_sparsedata+2, dtype=torch.int64)
            x,y = torch.meshgrid(x[1:-1], y[1:-1], indexing='ij')
            self.sparse_data_idx = x + y*self.args.nx
            self.sparse_data_idx = self.sparse_data_idx.view(-1,)
                


    def CLSTMNet(self, timearray,iterBack):
        args = self.args
        # para    = [30, 575e3]
        para    = [torch.abs(self.paranom[i])*self.paraOrder[i] for i in range(len(self.paraOrder))]

        timemin_save, timemin_print = args.save_dtmin, 1e-3

        timemin = torch.tensor(0).to(self.args.device)
        DOC     = (torch.ones((self.args.ncases, *self.grid_shape()))*0.005).to(self.args.device)
        theta   = (torch.ones((self.args.ncases, *self.grid_shape()))*self.getT_BC(0)).to(self.args.device)
        self.Data = {"theta0":[], "DOC0":[], "timemin0":[]}
            
        DocMean = self.d_DOC_mean
        DoCStd  = self.d_DOC_std
        Tmean   = self.d_theta_mean
        Tstd    = self.d_theta_std

        T_out = torch.zeros_like(theta).to(self.args.device)
        
        x = torch.cat([(theta[:,None]-Tmean)/Tstd,(DOC[:,None]-DocMean)/DoCStd], dim=1).unsqueeze(0)
        h = [None]*self.ConvLSTM.num_layers
        
        for i,timesec  in enumerate(timearray):
            timemin = (timesec/60.).to(self.args.device)
            for cases in range(self.args.ncases):
                T_out[cases] = self.getT_BC(timemin,cases)
            
            # x = torch.cat([(self.d_theta[i:i+1,:,None]-Tmean)/Tstd,(self.d_DOC[i:i+1,:,None]-DocMean)/DoCStd], dim=2).detach()
            x = torch.cat([(T_out[None,:,None]-Tmean)/Tstd,x], dim=2)
            x, h=self.ConvLSTM(x, h)
            
            theta_t = x[:,:,0]*Tstd+Tmean
            DOC     = x[:,:,1]*DoCStd+DocMean
            theta = self.update_BC_theta(timemin,theta_t[-1],torch.zeros_like(DOC[-1]),para,BC='convective')
            x[:,:,0] = (theta[None]-Tmean)/Tstd

            Datacurr = {"theta"+str(iterBack):theta[None].clone(), 
                        "DOC"+str(iterBack):DOC.clone(), 
                        "timemin"+str(iterBack):timemin[None].clone()}
            for Var in Datacurr:
                self.Data[Var].append(Datacurr[Var])

            if(timemin >= (args.nmins-args.dt/60.)): break
        
        for Var in Datacurr:
            self.Data[Var] = torch.cat(self.Data[Var], dim=0)


    def train_CLSTMNet(self):

        lossfu = nn.MSELoss()
        self.train_for_n_steps = 1

        ntimes    = int(self.args.nmins*60./self.args.dt)
        timearray = np.arange(ntimes)*self.args.dt
        timearray = torch.from_numpy(timearray).float()
        backsplit = [0, ntimes]
	
        self.optimizer[0].zero_grad()

        iterBack = 0
        itimes = min(((self.epoch)//self.train_for_n_steps+1)*1 ,ntimes-1)
        self.forward(timearray[:itimes],iterBack)

        pred_DOC   = torch.reshape(self.Data["DOC"  +str(iterBack)], (-1,self.args.ncases,self.args.nx*self.args.ny))
        pred_theta = torch.reshape(self.Data["theta"+str(iterBack)], (-1,self.args.ncases,self.args.nx*self.args.ny))
        pred_DOC   = pred_DOC[:,:,self.sparse_data_idx]
        pred_theta = pred_theta[:,:,self.sparse_data_idx]
        trgt_DOC   = self.d_DOC[:itimes,:,self.sparse_data_idx]
        trgt_theta = self.d_theta[:itimes,:,self.sparse_data_idx]

        ## Loss
        reg_loss  = self.args.loss_flg[2] *sum(p.abs().sum() for p in self.para_list)
        seq_loss  = self.args.loss_flg[3] *lossfu(self.Data["DOC"+str(iterBack)][1:,:], self.Data["DOC"+str(iterBack)][:-1,:])
        seq_loss += self.args.loss_flg[4] *lossfu(self.Data["theta"+str(iterBack)][1:,:], self.Data["theta"+str(iterBack)][:-1,:])
        loss_doc  = self.args.loss_flg[0] *lossfu(pred_DOC, trgt_DOC) /torch.max(torch.max(torch.max(torch.max(self.d_DOC))))
        loss_tem  = self.args.loss_flg[1] *lossfu(pred_theta, trgt_theta) /torch.max(torch.max(torch.max(torch.max(self.d_theta))))
        loss_con  = min(self.epoch, 200)*lossfu(torch.maximum(pred_DOC,torch.ones_like(pred_DOC)*1.1)-torch.ones_like(pred_DOC)*1.1, torch.zeros_like(pred_DOC))
        loss      = reg_loss + seq_loss + loss_doc + loss_tem + loss_con
        print("epoch = {}".format(self.epoch),"(",iterBack,")" " loss = {:.2f}".format(loss_doc.item()*1e3), 
                    "{:.2f}".format(loss_tem.item()*1e3),  "{:.5f}".format(reg_loss.item()*1e3), 
                    "{:.5f}".format(seq_loss.item()*1e3), "x1e-3")

        loss.backward()
        # nn.utils.clip_grad_norm_(self.para_list,1) 
        self.optimizer[0].step()
        self.optimizer[0].zero_grad()
        loss_item = loss.item()
        # print("para = ",para[0].item(), para[1].item())
        del loss, seq_loss, reg_loss, pred_DOC, pred_theta

        self.Data["theta"]=[]
        self.Data["DOC"]=[]
        self.Data["timemin"]=[]
        for Var in ["theta","DOC","timemin"]:
            self.Data[Var].append(self.Data[Var+str(iterBack)].detach().cpu())
            del self.Data[Var+str(iterBack)]
            self.Data[Var] = torch.cat(self.Data[Var], dim=0)

        return loss_item


    def Hybrid_solver(self, timearray,iterBack):
        args = self.args
        # para    = [30, 575e3]
        # para    = [(torch.abs(self.paranom[i])+torch.randn((1,))*self.optimizer[1].param_groups[0]['lr']*0.1)*self.paraOrder[i] for i in range(len(self.paraOrder))]
        para    = [(torch.abs(self.paranom[i]))*self.paraOrder[i] for i in range(len(self.paraOrder))]

        # timemin_save, timemin_print = args.save_dtmin, 1e-3

        if iterBack == 0:
            timemin = torch.tensor(0).to(self.args.device)
            DOC     = (torch.ones((self.args.ncases, *self.grid_shape()))*0.005).to(self.args.device)
            theta   = (torch.ones((self.args.ncases, *self.grid_shape()))*self.getT_BC(0)).to(self.args.device)
            self.Data = {"theta0":[], "DOC0":[], "timemin0":[]}
        else:
            DOC     = self.Data["DOC"+str(iterBack-1)][-1].detach()
            theta   = self.Data["theta"+str(iterBack-1)][-1].detach()
            timemin = self.Data["timemin"+str(iterBack-1)][-1].detach()
            self.Data["theta"+str(iterBack)]=[]
            self.Data["DOC"+str(iterBack)]=[]
            self.Data["timemin"+str(iterBack)]=[]
            
        for timesec  in timearray:
            timemin = timesec/60.
            DOC, DOCdot = self.stepintime_DOC(DOC,theta)
            theta = self.stepintime_theta(timemin,theta,DOCdot,para,BC='convective')

            # timemin = timemin + args.dt/60.

            # if (timemin > timemin_print):
            #     timemin_print = timemin + args.print_dtmin
            #     print("time =", "{:.2f}".format(timemin), "min, T =", "{:.2f}".format(K2Fahrenheit(self.getT_BC(timemin))),
            #     "F, DOC =", "{:.2f}".format(DOC[0,0,0]))

            # if (timemin > timemin_save):
            #     timemin_save = timemin + args.save_dtmin
                
            Datacurr = {"theta"+str(iterBack):theta[None].clone(), 
                        "DOC"+str(iterBack):DOC[None].clone(), 
                        "timemin"+str(iterBack):timemin[None].clone()}
            for Var in Datacurr:
                self.Data[Var].append(Datacurr[Var])

            if(timemin >= (args.nmins-args.dt/60.)): break
        
        for Var in Datacurr:
            self.Data[Var] = torch.cat(self.Data[Var], dim=0)

    
    # @profile
    def train_Hybrid_solver(self):

        lossfu = nn.MSELoss()

        ntimes     = int(self.args.nmins*60./self.args.dt)
        backsplit, backsplit_var = ntimes, 0
        backsplit  = np.rint(np.arange(1,ntimes//backsplit)*backsplit 
                    + np.random.normal(0., backsplit_var,ntimes//backsplit-1)).astype(int)
        timearrays = np.arange(ntimes)*self.args.dt
        backsplit  = np.append(backsplit, ntimes)
        timearrays = np.split(timearrays, backsplit)
        timearrays = [torch.from_numpy(item).float() for item in timearrays[:-1]]
        backsplit  = [0, *(backsplit)]
	
        self.optimizer[0].zero_grad()
        if self.args.train_paranorm: self.optimizer[1].zero_grad()
        
        for iterBack, timearray in enumerate(timearrays):
            self.forward(timearray,iterBack)

            pred_DOC   = torch.reshape(self.Data["DOC"  +str(iterBack)], (ntimes,self.args.ncases,self.args.nx*self.args.ny))
            pred_theta = torch.reshape(self.Data["theta"+str(iterBack)], (ntimes,self.args.ncases,self.args.nx*self.args.ny))
            pred_DOC   = pred_DOC[:,:,self.sparse_data_idx]
            pred_theta = pred_theta[:,:,self.sparse_data_idx]
            trgt_DOC   = self.d_DOC[:,:,self.sparse_data_idx]
            trgt_theta = self.d_theta[:,:,self.sparse_data_idx]

            ## Loss
            reg_loss  = self.args.loss_flg[2] *sum(p.abs().sum() for p in self.para_list)
            seq_loss  = self.args.loss_flg[3] *lossfu(self.Data["DOC"+str(iterBack)][1:,:], self.Data["DOC"+str(iterBack)][:-1,:])
            seq_loss += self.args.loss_flg[4] *lossfu(self.Data["theta"+str(iterBack)][1:,:], self.Data["theta"+str(iterBack)][:-1,:])
            loss_doc  = self.args.loss_flg[0] *lossfu(pred_DOC, trgt_DOC) /torch.max(torch.max(torch.max(self.d_DOC)))
            loss_tem  = self.args.loss_flg[1] *lossfu(pred_theta, trgt_theta) /torch.max(torch.max(torch.max(self.d_theta)))
            loss_con  = min(self.epoch, 200)*lossfu(torch.maximum(pred_DOC,torch.ones_like(pred_DOC)*1.1)-torch.ones_like(pred_DOC)*1.1, torch.zeros_like(pred_DOC))
            loss      = reg_loss + seq_loss + loss_doc + loss_tem + loss_con
            print("epoch = {}".format(self.epoch),"(",iterBack,")" " loss = {:.2f}".format((loss_doc.item()+loss_con.item())*1e3), 
                     "{:.2f}".format(loss_tem.item()*1e3),  "{:.5f}".format(reg_loss.item()*1e3), 
                     "{:.5f}".format(seq_loss.item()*1e3), "x1e-3",
                     "| para = {:.2f}, {:.2f}, {:.2f}, {:.2f}"
                     .format(self.paranom[0].item(),self.paranom[1].item(),self.paranom[2].item(),self.paranom[3].item()))

            loss.backward()
            # nn.utils.clip_grad_norm_(self.para_list,1) 
            self.optimizer[0].step()
            self.optimizer[0].zero_grad()
            if self.args.train_paranorm:
                self.optimizer[1].step()
                self.optimizer[1].zero_grad()
            loss_item = loss.item()
            # print("para = ",para[0].item(), para[1].item())
            del loss, seq_loss, reg_loss, pred_DOC, pred_theta

        self.Data["theta"]=[]
        self.Data["DOC"]=[]
        self.Data["timemin"]=[]
        for Var in ["theta","DOC","timemin"]:
            for iterBack in range(len(timearrays)):
                self.Data[Var].append(self.Data[Var+str(iterBack)].detach().cpu())
                del self.Data[Var+str(iterBack)]
            self.Data[Var] = torch.cat(self.Data[Var], dim=0)

        return loss_item
 

    def test(self):
        
        lossfu = nn.MSELoss()

        with torch.no_grad():

            ntimes     = int(self.args.nmins*60./self.args.dt)
            timearrays = np.arange(ntimes)*self.args.dt
            timearrays = [torch.from_numpy(timearrays).float()]
            backsplit  = [0, ntimes]
	
            for iterBack, timearray in enumerate(timearrays):
                self.forward(timearray,iterBack)

                pred_DOC   = torch.reshape(self.Data["DOC"  +str(iterBack)], (-1,self.args.ncases,self.args.nx*self.args.ny))
                pred_theta = torch.reshape(self.Data["theta"+str(iterBack)], (-1,self.args.ncases,self.args.nx*self.args.ny))
                pred_DOC   = pred_DOC[:,:,self.sparse_data_idx]
                pred_theta = pred_theta[:,:,self.sparse_data_idx]
                trgt_DOC   = self.d_DOC[:,:,self.sparse_data_idx]
                trgt_theta = self.d_theta[:,:,self.sparse_data_idx]

                ## Loss
                loss_avg_doc, loss_avg_tem = 0, 0
                for case in range(pred_DOC.size()[1]):
                    loss_doc = self.args.loss_flg[0] *lossfu(pred_DOC[:,case], trgt_DOC[:,case]) #/torch.max(torch.max(torch.max(self.d_DOC)))
                    loss_tem = self.args.loss_flg[1] *lossfu(pred_theta[:,case], trgt_theta[:,case]) #/torch.max(torch.max(torch.max(self.d_theta)))
                # reg_loss  = self.args.loss_flg[2] *sum(p.abs().sum() for p in self.para_list)
                # seq_loss  = self.args.loss_flg[3] *lossfu(self.Data["DOC"+str(iterBack)][1:,:], self.Data["DOC"+str(iterBack)][:-1,:])
                # seq_loss += self.args.loss_flg[4] *lossfu(self.Data["theta"+str(iterBack)][1:,:], self.Data["theta"+str(iterBack)][:-1,:])
                # loss += reg_loss + seq_loss
                    loss = loss_doc + loss_tem
                    print("case = {}".format(case), " test loss = {:.2f}".format(loss_doc.item()*1e3), "{:.2f}".format(loss_tem.item()*1e3), "x1e-3")
                    loss_avg_doc += loss_doc.item()
                    loss_avg_tem += loss_tem.item()
                print("Avg loss = ", loss_avg_doc*1e3/pred_DOC.size()[1], loss_avg_tem*1e3/pred_DOC.size()[1], "x1e-3")

            self.Data["theta"]=[]
            self.Data["DOC"]=[]
            self.Data["timemin"]=[]
            for Var in ["theta","DOC","timemin"]:
                for iterBack in range(len(timearrays)):
                    self.Data[Var].append(self.Data[Var+str(iterBack)].detach().cpu())
                    del self.Data[Var+str(iterBack)]
                self.Data[Var] = torch.cat(self.Data[Var], dim=0)

            # scheduler.step(loss.item())
            # plot_valdation(self,self.Data,self.Data_Exp,cases=[0], dir=self.args.pred_dir,name="testing_plot_epoch{}".format(self.epoch))

            for case in range(self.args.ncases):
                plot_valdation(self,self.Data,self.Data_Exp,cases=[case], dir=self.args.val_dir,name="validation_plot_case{}".format(case))
   

    def training_loop(self):

        if self.args.epochstart == 0:
            Train_loss_file = open(self.args.out_dir+"/Train_loss.txt","w")
            Test_loss_file  = open(self.args.out_dir+"/Test_loss.txt","w")
            para_file       = open(self.args.out_dir+"/paranorm.txt","w")
            paranom_txt= [self.paranom[i].item() for i in range(len(self.paranom))]
            para_file.write(str(0)+'\t'+'\t'.join(map(str,paranom_txt))+'\n')
        else:
            Train_loss_file = open(self.args.out_dir+"/Train_loss.txt","a")
            Test_loss_file  = open(self.args.out_dir+"/Test_loss.txt","a")
            para_file       = open(self.args.out_dir+"/paranorm.txt","a")

        # with torch.profiler.profile(
        #     activities=[torch.profiler.ProfilerActivity.CPU,
        #                 torch.profiler.ProfilerActivity.CUDA],
        #     # schedule=torch.profiler.schedule(wait=0, warmup=0, active=2),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./result_tb1'),
        #     record_shapes=True,
        #     # profile_memory=True,
        #     # with_stack=True
        #     ) as prof:

        for self.epoch in range(self.args.epochstart+1, self.args.nepoch+1):
            
            train_loss = self.train()
            Train_loss_file.write(str(self.epoch)+'\t'+str(train_loss)+'\n')
            paranom_txt= [abs(self.paranom[i].item()) for i in range(len(self.paranom))]
            para_file.write(str(self.epoch)+'\t'+'\t'.join(map(str,paranom_txt))+'\n')
            # para_file.write(str(self.epoch)+paranom_txt+'\n')
            
            # if self.epoch%10 == 0:
            #     test_loss = self.test(paranom, self.epoch)
            #     Test_loss_file.write(str(self.epoch+1)+'\t'+str(test_loss)+'\n')

            if(self.args.train_CLSTM):
                if(self.epoch>int(self.args.nmins*60./self.args.dt)*self.train_for_n_steps):
                    self.scheduler[0].step(train_loss)
            else:
                self.scheduler[0].step(train_loss)
                if self.args.train_paranorm:
                    self.scheduler[1].step(train_loss)

            if(self.epoch%self.args.save_model_depoch == 0):
                self.saveModel(self.epoch, self.optimizer, self.scheduler)
            
            # prof.step()
            
            if(self.epoch%self.args.plot_train_depoch == 0):
                plot_valdation(self,self.Data,self.Data_Exp,cases=[0], dir=self.args.pred_dir,name="training_plot_epoch{}".format(self.epoch))
            # plot_valdation(self,self.Data,self.Data_Exp,cases=[0], dir=self.args.pred_dir,name="training_plot_epoch{}".format(0))

        Train_loss_file.close()
        Test_loss_file.close()
        para_file.close()


    def gen_Data(self,name="testing_data"):
    
        with torch.no_grad():

            ntimes = int(self.args.nmins*60./self.args.dt)
            timearray = torch.from_numpy(np.arange(ntimes)*self.args.dt)
	
            self.Hybrid_solver(timearray,0)
            
            for Var in ["theta","DOC","timemin"]:
                self.Data[Var] = self.Data.pop(Var+str(0))
                self.Data[Var] = self.Data[Var].detach().cpu()

        self.save_states(self.Data,name)


    def save_states(self, Data, name="traning_data"):
        print("Saving state variables")
        with h5py.File(self.args.in_dir+"/"+name+".hdf5", "w") as hdf:
            Gphi = hdf.create_group("Gphi")
            Gphi.create_dataset("timeminsBC",  data= self.args.timeminsBC)
            Gphi.create_dataset("thetavalBC",  data= self.args.thetavalBC)
            Gphi.create_dataset("theta",  data= Data["theta"])
            Gphi.create_dataset("DOC",    data= Data["DOC"])
            Gphi.create_dataset("timemin",data= Data["timemin"])
   
    def load_states(self, name="traning_data"):
        print("loading state variables")
        Data = {}
        with h5py.File(self.args.in_dir+"/"+name+".hdf5", "r") as hdf:
            self.args.timeminsBC  = torch.tensor(np.array(hdf["Gphi"]["timeminsBC"]))
            self.args.thetavalBC  = torch.tensor(np.array(hdf["Gphi"]["thetavalBC"]))
            Data["theta"]  = torch.tensor(np.array(hdf["Gphi"]["theta"]))
            Data["DOC"]    = torch.tensor(np.array(hdf["Gphi"]["DOC"]))
            Data["timemin"]= torch.tensor(np.array(hdf["Gphi"]["timemin"]))

            self.args.ncases = len(self.args.timeminsBC)
            self.args.nmins  = self.args.timeminsBC[0,-1]

            # theta   = Data["theta"][-1]
            # DOC     = Data["DOC"][-1]
            # timemin = Data["timemin"][-1]

        return Data#, theta,DOC,timemin

    def saveModel(self, epoch, optimizer, scheduler):
        '''
        Save neural network
        '''
        print('Epoch = {}, Saving model!'.format(epoch))
        # Create state dict of both the model and optimizer
        state = {'epoch': epoch,
                'optimizer_n': optimizer[0].state_dict(), 
                'scheduler_n': scheduler[0].state_dict(), 
                'optimizer_p': optimizer[1].state_dict(), 
                'scheduler_p': scheduler[1].state_dict(), 
                # 'optimizer': optimizer, 
                # 'scheduler': scheduler, 
                'DOC_mean': self.d_DOC_mean, 
                'DOC_std': self.d_DOC_std, 
                'theta_mean': self.d_theta_mean, 
                'theta_std': self.d_theta_std, 
                'time_std': self.d_time_std, 
                'paranom': self.paranom}
        if self.args.addNNDOC:
            state['state_dict_NNDOC'] = self.get_DOCdot_NN.state_dict()
        if self.args.addNNEng:
            state['state_dict_NNtheta'] = self.get_thetadot_NN.state_dict()
        if self.args.train_CLSTM:
            state['state_dict_CLSTM'] = self.ConvLSTM.state_dict()
        torch.save(state, self.args.ckpt_dir+'/Model_epoch{:d}.pth'.format(epoch))

    def loadModel(self, epoch, optimizer=None, scheduler=None):
        '''
        Loads pre-trained network from file
        '''
        # try:
        file_name = self.args.ckpt_dir+'/Model_epoch{:d}.pth'.format(epoch)
        param_dict = torch.load(file_name, map_location=lambda storage, loc: storage)
        print('Found model at epoch: {:d}'.format(param_dict['epoch']))
        # except FileNotFoundError:
        #     print('Error: Could not find PyTorch network')
        #     return
        
        # Load NN
        if self.args.addNNDOC:
            self.get_DOCdot_NN.load_state_dict(param_dict['state_dict_NNDOC'])
        if self.args.addNNEng:
            self.get_thetadot_NN.load_state_dict(param_dict['state_dict_NNtheta'])
        if self.args.train_CLSTM:
            self.ConvLSTM.load_state_dict(param_dict['state_dict_CLSTM'])
        # Load optimizer/scheduler
        if(not optimizer is None):
            optimizer[0].load_state_dict(param_dict['optimizer_n'])
            scheduler[0].load_state_dict(param_dict['scheduler_n'])
            optimizer[1].load_state_dict(param_dict['optimizer_p'])
            scheduler[1].load_state_dict(param_dict['scheduler_p'])
            # optimizer = param_dict['optimizer']
            # scheduler = param_dict['scheduler']
        self.d_DOC_mean   = param_dict['DOC_mean']
        self.d_DOC_std    = param_dict['DOC_std']
        self.d_theta_mean = param_dict['theta_mean']
        self.d_theta_std  = param_dict['theta_std']
        self.d_time_std   = param_dict['time_std']
        self.paranom.data = param_dict['paranom']
        print('Pre-trained model loaded!')

        return optimizer, scheduler