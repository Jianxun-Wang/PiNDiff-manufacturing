import sys
import time
import datetime
import torch
from torch import nn
from torch.optim import lr_scheduler
from types import SimpleNamespace
import yaml
import tracemalloc

from solver.solver import *
from utils.utils import EmbeddingParser, K2Fahrenheit
from utils.post import *

from solver.NN_solver.phase_NNsolver import Phase_NNsolver, Phase_INsolver
from solver.NN_solver.energy_NNsolver import Energy_NNsolver

starttime = time.time()
tracemalloc.start()

def write_performance_file(args):

    perfomance = open(args.out_dir+"/performance_"+str(args.nepoch)+".txt","w")
    perfomance.write('\tepoch start =\t\t\t'+str(args.epochstart)+'\n')
    perfomance.write('\tno. of epochs =\t\t'+str(args.nepoch)+'\n')
    perfomance.write('\tno. of parameters =\t\t'+str( pytorch_total_params/1e3)+' k\n')
    perfomance.write('\tpeak memory consumption =\t'+str(format(tracemalloc.get_traced_memory()[1]/(1024*1024), ".2f"))+' Gbs \n')
    perfomance.write('\ttotal run time =\t\t'+str(time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)))+'\n\n')

    perfomance.write("Model parameters:\n")
    if args.addNNDOC: 
        perfomance.write("\tANN for DOC:"+" layers = "+str(args.NNDOC_layers)+" hsize = "+str(args.NNDOC_hsize)+'\n')
    if args.addNNEng: 
        perfomance.write("\tCNN for T:"+" layers = "+str(args.NNEng_layers)+" hsize = "+str(args.NNEng_hsize)+'\n')
    if args.train_CLSTM: 
        perfomance.write("\tConv-LSTM:"+" hsize = "+str(args.CLSTM_hsize)+'\n')

    perfomance.write("\tLoss   = "+str(args.loss_flg)+'\n')
    perfomance.write("\tnx, ny = "+str(args.nx)+', '+str(args.nx)+'\n')
    perfomance.write("\tlx, ly = "+str(args.lx[1])+', '+str(args.lx[1])+'\n')
    perfomance.write("\tdt     = "+str(args.dt)+'\n')
    perfomance.write("\ttarget_sparsedata = "+str(args.target_sparsedata)+'\n')

    perfomance.write("\nDate and time:\n")
    perfomance.write('\t'+str(datetime.datetime.now()))
    perfomance.close()


def print_plan(args):

    if args.gen_syndata:
        if args.train_model: print("Generating sysnthetic traning data")
        else: print("Generating sysnthetic testing data")

    else:
        if args.train_model: print("Traning model with:")
        else: print("Testing model with:")
        if args.addNNDOC: print(" -Artifical neural network for degree of cure:", 
                                "layers = ",args.NNDOC_layers,"hsize = ",args.NNDOC_hsize)
        if args.addNNEng: print(" -Convolutional neural network for degree of Energy equation:", 
                                "layers = ",args.NNEng_layers,"hsize = ",args.NNEng_hsize)
        if args.train_CLSTM: print(" -Convolution LSTM network:" ,"hsize = ", args.CLSTM_hsize)
        if args.train_model: print("loss=",args.loss_flg )


############################################################################################################

#device configuration 
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# torch.set_num_threads(2)

args = EmbeddingParser().parse()  

# args.train_model = True
# args.gen_syndata = True
# args.gen_rnd_batch = True
# args.train_paranorm = True
rnd_batch_size = 50


if not args.gen_syndata:
    # args.epochstart = 0
    # args.nepoch = 300
    # args.addNNDOC = True
    # args.addNNEng = True
    # args.train_CLSTM = True
    # args.loss_flg = [0, 1, 1e-8, 1, 1e-5]     ##### [loss(DOC), loss(T), loss(w), loss(DOC(i+1)-DOC(i)), loss(T(i+1)-T(i))]

    args.loss_flg = [float(item) for item in args.loss_flg.split(',')]
else:
    if args.train_model:
        file_params = SimpleNamespace(**yaml.load(open(args.in_dir+"/training_case.yaml"), Loader=yaml.FullLoader))
    else:
        file_params = SimpleNamespace(**yaml.load(open(args.in_dir+"/testing_case.yaml"), Loader=yaml.FullLoader))

    if not args.gen_rnd_batch:
        args.timeminsBC = torch.tensor(file_params.timeminsBC)
        args.thetavalBC = torch.tensor(file_params.thetavalBC)
    else:
        args.timeminsBC = torch.tensor([file_params.timeminsBC[0] for i in range(rnd_batch_size)])
        args.thetavalBC = torch.cat((65.*torch.ones(rnd_batch_size,1),275.+ 25*(2*torch.rand(rnd_batch_size,6)-1),65.*torch.ones(rnd_batch_size,1)), dim=1)

    args.ncases = len(args.timeminsBC)
    args.nmins  = args.timeminsBC[0,-1]

print_plan(args)

true_paranom = torch.tensor([4.,4.2,3.302,0.5067], device=args.device)
if args.gen_syndata or not args.train_paranorm:
    paranom = true_paranom
else:
    paranom = nn.Parameter(torch.tensor([1., 1., 1., 1.], device=args.device), requires_grad=True)
paraOrder = [10, 1e5, 1, 1]


############################################################################################################


if args.train_model:
    ccc = cccSolver(args, name="traning_data")
else:
    ccc = cccSolver(args, name="testing_data")

ccc.paranom   = paranom
ccc.paraOrder = paraOrder

if args.gen_syndata:
    if args.train_model:
        ccc.gen_Data(name="traning_data")
    else:
        ccc.gen_Data(name="testing_data")
else:
    
    ccc.para_list = []
    pytorch_total_params = 0
    if args.addNNDOC:
        ccc.get_DOCdot_NN = Phase_NNsolver(args).to(args.device)
        # ccc.get_DOCdot_NN = Phase_INsolver(args).to(args.device)
        for parameters in ccc.get_DOCdot_NN.parameters(): ccc.para_list.append(parameters)
        pytorch_total_params += sum(p.numel() for p in ccc.get_DOCdot_NN.parameters() if p.requires_grad)
    if args.addNNEng:
        ccc.get_thetadot_NN = Energy_NNsolver(args).to(args.device)
        for parameters in ccc.get_thetadot_NN.parameters(): ccc.para_list.append(parameters)
        pytorch_total_params += sum(p.numel() for p in ccc.get_thetadot_NN.parameters() if p.requires_grad)

    if args.train_CLSTM:
        ccc.ConvLSTM = ConvLSTM_Net(args, nu_inp_t=1, in_channels=3, out_channels=2, LSTM_hc= args.CLSTM_hsize).to(args.device)
        for parameters in ccc.ConvLSTM.parameters(): ccc.para_list.append(parameters)
        pytorch_total_params += sum(p.numel() for p in ccc.ConvLSTM.parameters() if p.requires_grad)

    ccc.optimizer = [torch.optim.Adam(ccc.para_list, lr=args.lr),torch.optim.SGD([ccc.paranom], lr=1)]
    ccc.scheduler = [lr_scheduler.ReduceLROnPlateau(ccc.optimizer[0], mode='min', factor=0.7, patience=20, cooldown=0, min_lr=1e-4, verbose=True),
                     lr_scheduler.ReduceLROnPlateau(ccc.optimizer[1], mode='min', factor=0.7, patience=20, cooldown=0, min_lr=1e-4, verbose=True)]

    print("no of parameters = ", pytorch_total_params/1e3, "k")

    if args.epochstart > 0:
        ccc.optimizer, ccc.scheduler = ccc.loadModel(args.epochstart, ccc.optimizer, ccc.scheduler)


    if args.train_model:
        starttime = time.time()
        ccc.training_loop()
        plot_loss(ccc)
        if args.train_paranorm: plot_para(ccc, true_paranom.cpu().numpy())

    else:
        starttime = time.time()
        ccc.test()


endtime = time.time()
print("time = ",time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)))

if (not args.gen_syndata) and args.train_model:
    write_performance_file(args)
    
tracemalloc.stop()


plotAlphaT_withtime(ccc,ccc.Data["timemin"],ccc.Data["theta"],ccc.Data["DOC"],cases=[0], idxlist=[[5,5]])
# plot_Var(ccc, ccc.Data["timemin"],ccc.Data["theta"], cases=[0], tsteps=20, Var_name='temperature')
# plot_Var(ccc, ccc.Data["timemin"],ccc.Data["DOC"], cases=[0], tsteps=20, Var_name='DOC')
# plot_2DVar(ccc,ccc.Data,ccc.Data_Exp, cases=[0], tsteps=8, Var_name='theta')
# plot_2DVar(ccc,ccc.Data,ccc.Data_Exp, cases=[0], tsteps=8, Var_name='DOC')