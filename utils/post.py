from math import floor
import os, errno
import shutil
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.utils import printProgressBar, K2Fahrenheit

# style1 = ['-r','--r','^r','+r','*r']
# style2 = ['--r','--g','^g','+g','*g']

style1 = ['-r','-g','-b','-c','-m']
style2 = ['--r','--g','--b','--c','--m']

def plotAlphaT_withtime(ccc,timemin,theta,DOC,cases=[0],idxlist=[[0,0]]):
    
    plt.close("all")
    fig,ax1 = plt.subplots()
    ax2=ax1.twinx()

    # T_out = np.zeros((len(timemin[1:])))
    # for i in range(len(timemin[1:])):
    #     T_out[i] = K2Fahrenheit(ccc.getT_BC(timemin[i+1],case))
    # ax2.plot(timemin[1:],T_out,'-k', label="T oven")

    ax2.set_ylabel("Temperature, F")
    
    for i, idx in enumerate(idxlist):
        for j, case in enumerate(cases):
            ax2.plot(timemin[1:], K2Fahrenheit(theta[1:,case,idx[0],idx[1]]),style1[i*len(idxlist)+j], label="Temperature")
            ax1.plot(timemin[1:], DOC[1:,case,idx[0],idx[1]],style2[i*len(idxlist)+j], label="DOC")

    ax1.legend( bbox_to_anchor=(0.7, 0.3))
    # ax2.legend( bbox_to_anchor=(0.2, 0.68))
    ax1.set_xlabel("Time, mins")
    ax1.set_ylabel("DOC")
    ax2.grid()
    plt.savefig(ccc.args.plot_dir+"/DOC_theta_t.pdf")


def plot_valdation(ccc,Data,DataE,cases,dir,name):
    
    t_len = len(Data["theta"])

    plt.close("all")
    fig,ax = plt.subplots(2, 1, figsize=(1*5, 2*4))
    plt.subplots_adjust(wspace=0.0,hspace=0.0)
    plt.rcParams['font.size'] = '18'
    # ax2=ax1.twinx()

    T_out = np.zeros((len(Data["timemin"][0:])))
    for i in range(len(Data["timemin"][0:])):
        T_out[i] = (ccc.getT_BC(Data["timemin"][i], cases[0]))
    ax[0].plot(Data["timemin"][0:],T_out,'-k', label="Autoclave temperature", linewidth=1)

    idx = [round(ccc.args.nx/2), round(ccc.args.ny/2)]
    for j, case in enumerate(cases):
        ax[0].plot( Data["timemin"][0:], (Data["theta"][0:,case,idx[0],idx[1]]),style1[j+0], label="T centre")
        ax[0].plot(DataE["timemin"][0:t_len], (DataE["theta"][0:t_len,case,idx[0],idx[1]]),style2[j+0])
        ax[0].plot( Data["timemin"][0:], (Data["theta"][0:,case,1,1]),style1[j+2], label="T edge")
        ax[0].plot(DataE["timemin"][0:t_len], (DataE["theta"][0:t_len,case,1,1]),style2[j+2])

        ax[1].plot( Data["timemin"][0:], (Data["DOC"][0:,case,idx[0],idx[1]]),style1[j], label="DOC centre")
        ax[1].plot(DataE["timemin"][0:t_len], (DataE["DOC"][0:t_len,case,idx[0],idx[1]]),style2[j])
        ax[1].plot( Data["timemin"][0:], (Data["DOC"][0:,case,1,1]),style1[j+2], label="DOC edge")
        ax[1].plot(DataE["timemin"][0:t_len], (DataE["DOC"][0:t_len,case,1,1]),style2[j+2])

    ax[0].set_xlim([-10, 420])
    ax[0].set_ylim([280, 510])
    ax[1].set_xlim([-10, 420])
    ax[1].set_ylim([-0.1, 1.1])
    ax[0].grid()
    ax[1].grid()
    # ax[0].legend()
    # ax[1].legend()
    # ax[0].set_xlabel("Time (mins)")
    ax[1].set_xlabel("Time (mins)")
    ax[0].set_ylabel("Temperature (K)")
    ax[1].set_ylabel("Degree of cure")
    # ax[0].tick_params(left = False, labelleft = False)
    # ax[1].tick_params(left = False, labelleft = False)
    plt.savefig(dir+"/"+name+".pdf", bbox_inches='tight',dpi=300,pad_inches = 0.02)

    
def plotrhoT(ccc,timemin,theta,rho_rho0,idx):
    
    plt.close("all")
    fig,ax1 = plt.subplots()

    # ax2=ax1.twinx()
    # ax2.plot(timemin[1:], (theta[1:,idx,idx]-375.)/1000.,'--k', label="Temperature")
    # ax2.set_ylabel("Temperature, x10^3 degree")

    ax1.plot(K2Fahrenheit(theta[1:,idx,idx]), rho_rho0[1:,idx,idx], label="0.1 K/s")

    ax1.legend( bbox_to_anchor=(0.3, 0.88))
    # ax2.legend( bbox_to_anchor=(0.2, 0.68))
    ax1.set_xlabel("Temperature, degree")
    ax1.set_ylabel("rho/rho0")
    plt.savefig(ccc.args.plot_dir+"/rho_T.pdf")


def plot_2DVar_sep(ccc, timemin, Var, cases, tsteps=10, Var_name='temperature'):
    '''
    Plots several timesteps
    '''
    for j, case in enumerate(cases):

        pred_dir = ccc.args.plot_dir +"/"+ Var_name+"_case"+str(case)

        try:
            shutil.rmtree(pred_dir)
        except:
            print(Var_name+"directory not deleted")

        try:
            os.makedirs(pred_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # Create figure
        mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
        mpl.rc('text', usetex=False)

        for i in range(tsteps+1):
            plt.close("all")
            fig, ax = plt.subplots(1, 1, figsize=(0.6*ccc.args.nx, 0.5*ccc.args.ny))
            # fig, ax = plt.subplots(1, 1, figsize=(5,5))
            # timeidx = sum(timemin<(ccc.nmins/tsteps *i))
            timeidx = round(len(timemin)/tsteps)*i
            if timeidx >= len(timemin):
                timeidx = len(timemin)-1
            time = timemin[timeidx]
            # ax.set_title(f'time={time:.3f}')

            ax0 = ax
            ax0.axis('off')
            # cmap = 'viridis' if i_ax in [2, 5] else 'plasma'

            vmax = np.max(np.array( Var[timeidx,case] ))
            vmin = np.min(np.array( Var[timeidx,case] ))
            # vmax = max(ccc.args.thetavalBC)#np.max(np.array( theta[timeidx] ))
            # vmin = min(ccc.args.thetavalBC)#np.min(np.array( theta[timeidx] ))
            cax = ax0.contourf(np.rot90(np.flipud(Var[timeidx,case]),-1), vmin=vmin, vmax=vmax)

            cbar = plt.colorbar(cax, ax=ax0, fraction=0.046, pad=0.04, 
                format=mpl.ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.update_ticks()

            ax0.set_title(Var_name)
            
            plt.tight_layout()
            plt.savefig(pred_dir + f'/{Var_name}_time={time:.1f}.pdf', bbox_inches='tight')
            plt.close()
            printProgressBar(i,tsteps,"Plotting "+Var_name)


def plot_2DVar(ccc, Data,DataE, cases, tsteps=10, Var_name='theta'):
    '''
    Plots several timesteps
    '''
    timemin = Data["timemin"]

    for j, case in enumerate(cases):

        pred_dir = ccc.args.plot_dir 
        
        # Create figure
        mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
        mpl.rc('text', usetex=False)

        plt.close("all")
        fig, ax = plt.subplots(tsteps, 3, figsize=(3*0.6*ccc.args.nx, tsteps*0.5*ccc.args.ny))

        for i in range(1,tsteps+1):
            # fig, ax = plt.subplots(1, 1, figsize=(5,5))
            # timeidx = sum(timemin<(ccc.nmins/tsteps *i))
            timeidx = round(len(timemin)/tsteps)*i
            if timeidx >= len(timemin):
                timeidx = len(timemin)-1
            time = timemin[timeidx]
            # ax.set_title(f'time={time:.3f}')

            for j in range(0,3):
                ax0 = ax[i-1,j]
                ax0.axis('off')

                if Var_name=='theta':
                    vmax, vmin = 470., 290.
                    emax, emin = 1., 0.
                    nspace = 10
                elif Var_name=='DOC':
                    vmax, vmin = 1., 0.
                    emax, emin = 10., 0.
                    nspace = 11

                if j == 0:
                    cax = ax0.contourf(np.rot90(np.flipud(Data[Var_name][timeidx,case]),-1),levels = np.linspace(vmin,vmax,nspace), vmin=vmin, vmax=vmax)
                    ax0.set_title(f'{time:.0f} min', rotation='vertical',x=-0.02,y=0.05, fontsize=140)
                elif j == 1:
                    cax = ax0.contourf(np.rot90(np.flipud(DataE[Var_name][timeidx,case]),-1),levels = np.linspace(vmin,vmax,nspace), vmin=vmin, vmax=vmax)
                else:
                    cax = ax0.contourf(np.rot90(np.flipud(100.*np.abs(Data[Var_name][timeidx,case]-DataE[Var_name][timeidx,case])/
                    DataE[Var_name][timeidx,case]),-1),levels = np.linspace(emin,emax,11), vmin=emin, vmax=emax, extend='max')
                
                # ax0.set_title(Var_name)
                printProgressBar(i,tsteps,"Plotting "+Var_name)
                
                # plt.tight_layout()
                # plt.savefig(pred_dir + f'/{Var_name}_time={time:.1f}.pdf', bbox_inches='tight')
                # plt.close()
        
        plt.tight_layout()
        plt.savefig(pred_dir + f'/{Var_name}.pdf', bbox_inches='tight')
        plt.close()


def plot_phi(ccc, timemin, phi, tsteps=10):
    '''
    Plots several timesteps
    '''
    pred_dir = ccc.args.plot_dir_phi

    # Create figure
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    mpl.rc('text', usetex=False)

    plttitles = [f'phi{i+1}' for i in range(9)]
    for i in range(tsteps+1):
        plt.close("all")
        fig, ax = plt.subplots(3, 3, figsize=(3*3.5, 3*3))
        # timeidx = sum(timemin<(ccc.args.nmins/tsteps *i))
        timeidx = floor((len(timemin))/tsteps)*i
        if timeidx >= len(timemin):
            timeidx = len(timemin)-1
        time = timemin[timeidx]
        # ax.set_title(f'time={time:.3f}')

        for i_ax in range(3):
            for j_ax in range(3):
                ax0 = ax[i_ax, j_ax]
                ax0.axis('off')
                # cmap = 'viridis' if i_ax in [2, 5] else 'plasma'

                vmax = np.max(np.array( phi[timeidx,3*i_ax+j_ax] ))
                vmin = np.min(np.array( phi[timeidx,3*i_ax+j_ax]  ))
                cax = ax0.contourf(np.flipud(phi[timeidx,3*i_ax+j_ax] ), vmin=vmin, vmax=vmax)

                cbar = plt.colorbar(cax, ax=ax0, fraction=0.046, pad=0.04, 
                    format=mpl.ticker.ScalarFormatter(useMathText=True))
                cbar.formatter.set_powerlimits((-2, 2))
                cbar.update_ticks()

                ax0.set_title(plttitles[3*i_ax+j_ax])
        
        plt.tight_layout()
        plt.savefig(pred_dir + f'/phi_time={time:.1f}.pdf', bbox_inches='tight')
        plt.close()
        printProgressBar(i,tsteps,"Plotting phi")


def plot_loss(ccc):
    
    Train_loss_file = open(ccc.args.out_dir+"/Train_loss.txt","r")
    
    lines = Train_loss_file.readlines()
    Train_loss = np.zeros((len(lines),2))
    
    for idx, line in enumerate(lines):
        sline = line.strip()
        sline = sline.split('\t')
        Train_loss[idx,0] = int(sline[0])
        Train_loss[idx,1] = float(sline[1])
    
    plt.close("all")
    fig,ax1 = plt.subplots()

    # ax2=ax1.twinx()
    # ax2.plot(timemin[1:], (theta[1:,idx,idx]-375.)/1000.,'--k', label="Temperature")
    # ax2.set_ylabel("Temperature, x10^3 degree")

    ax1.plot(Train_loss[10:,0],Train_loss[10:,1], label="Traning Loss")

    ax1.legend( bbox_to_anchor=(0.3, 0.88))
    # ax2.legend( bbox_to_anchor=(0.2, 0.68))
    ax1.set_xlabel("nepoch")
    ax1.set_ylabel("Loss")
    plt.savefig(ccc.args.plot_dir+"/Loss.pdf", bbox_inches='tight')


def plot_para(ccc, true_paranom):
    
    Train_loss_file = open(ccc.args.out_dir+"/paranorm.txt","r")
    
    lines = Train_loss_file.readlines()
    sline = lines[0].strip()
    sline = sline.split('\t')
    paranorm = np.zeros((len(lines),len(sline)))
    truepara = np.zeros((len(lines),len(sline)))
    
    for idx, line in enumerate(lines):
        sline = line.strip()
        sline = sline.split('\t')
        paranorm[idx,0] = int(sline[0])
        truepara[idx,0] = int(sline[0])
        for i in range(len(sline)-1):
            paranorm[idx,i+1] = float(sline[i+1])
            truepara[idx,i+1] = true_paranom[i]
    
    plt.close("all")
    plt.rcParams['font.size'] = '14'

    plt.plot(paranorm[:,0],paranorm[:,1],'r', label="h")
    plt.plot(truepara[:,0],truepara[:,1],'--r')
    plt.plot(paranorm[:,0],paranorm[:,2],'g', label="Hu")
    plt.plot(truepara[:,0],truepara[:,2],'--g')
    plt.plot(paranorm[:,0],paranorm[:,3],'b', label="kxx")
    plt.plot(truepara[:,0],truepara[:,3],'--b')
    plt.plot(paranorm[:,0],paranorm[:,4],'m', label="kyy")
    plt.plot(truepara[:,0],truepara[:,4],'--m')

    # plt.legend()
    plt.xlabel("no. of epochs")
    plt.ylabel("Physical parameters value")
    plt.savefig(ccc.args.plot_dir+"/paranorm.pdf", bbox_inches='tight')

# # animation
# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro')

# def init():
#     # ax.set_xlim(-10, 10)
#     # ax.set_ylim(-10, 10)
#     return ln,

# def update(frame):
#     xdata.append(pred[frame,1])
#     ydata.append(pred[frame,2])
#     ln.set_data(xdata, ydata)
#     return ln,

# ani = FuncAnimation(fig, update, frames=range(niter),
#                     init_func=init, blit=True)
