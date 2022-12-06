import numpy as np
import matplotlib.pyplot as plt
import simpy
import seaborn as sns
from tqdm import tqdm


from Q_types import *

import os
if not os.path.exists('Data'):
    os.makedirs('Data')
if not os.path.exists('Figs'):
    os.makedirs('Figs')
if not os.path.exists('Data/waitingtimerho'):
    os.makedirs('Data/waitingtimerho')

##### Functions for generating simulations ########
def sim(n_cust,rho_lo, rho_hi, num_sims=50, num_rho=15, n_serv=True, FIFO_b=True,Det_b=False, LongT_b=False):

    """ Generate simulation for M/M/n queueing systems
        Args
            n_cust      number of customers
            rho_lo      minimum server load
            rho_hi      maximum server load
            num_sims    number of measuemrements
            num_rho     number of server load settings
            n_serv      boolean for multiple servers
            FIFO_b      boolean for using FIFO queue
            Det_b       boolean for using deterministic service times
            LongT_b     boolean for using Hyperexponential service times
        Returns
            waiting times and service times for given settings
        """

    np.random.seed(42)
    # running simulations for MM1, MM2, and MM4 queue (Except for SJF, we only use M/M/1)
    num_serv = [1,2,4]
    if n_serv==False:
        num_serv = [num_serv[0]]

    # generate increasing system loads
    rhos = np.linspace(rho_lo, rho_hi, num_rho)
    arrival_rates = [1,2,4]

    wait_times = np.array([np.zeros((len(rhos),num_sims,n_cust)),   # 1 server
                           np.zeros((len(rhos),num_sims,n_cust)),   # 2 servers
                           np.zeros((len(rhos),num_sims,n_cust))])  # 4 servers
    serv_times = np.array([np.zeros((len(rhos),num_sims,n_cust)),
                           np.zeros((len(rhos),num_sims,n_cust)),
                           np.zeros((len(rhos),num_sims,n_cust))])
    if n_serv==False:
        num_serv = [num_serv[0]]
        wait_times = np.array([wait_times[0]])
        serv_times = np.array([serv_times[0]])

    for i, n in enumerate(num_serv):
        arrival_rate = n
        
        for sim in range(num_sims):
            for r, rho in tqdm(enumerate(rhos)):
                mu = 1 / rho

                queue = Queues(arrival_intvl=1/arrival_rate, service_intvl=1/mu, n_serv=n, n_cust=n_cust,FIFO=FIFO_b,Det=Det_b,LongT=LongT_b)
                wait_times[i,r, sim, :], serv_times[i,r, sim, :] = queue.sim()

    
    # mean over the last 1000:-1000 waiting times for each rho, n, Nsim
    cust_mean_wait = np.mean(wait_times[:,:,:,1000:-1000], axis=3) # new dim = [n_serv,rho,sim_num]


    # Saving files
    fileName = 'MEAN_WAITS_rho_' + str(num_rho) + '_NSIM' + str(num_sims) + '_' +str(FIFO_b)+str(Det_b) + str(LongT_b)
    np.save('Data/waitingtimerho/'+fileName, cust_mean_wait)

    
    return wait_times, serv_times



##################### PLOTTING ################################################################################################################################################

def plotter(rho_lo=0.5, rho_hi=0.95, num_reps=50, num_rho=15, save = False, show=False, FIFO_b=True, Det_b=False, LongT_b=False, name="Doolittle"):

    """ Plot various distributions, system loads and server number waiting times
        Args
            rho_lo      minimum server load
            rho_hi      maximum server load
            num_reps    number of measuemrements
            num_rho     number of server load settings
            save        boolean for saving output
            FIFO_b      boolean for using FIFO queue
            Det_b       boolean for using deterministic service times
            LongT_b     boolean for using Hyperexponential service times
        Returns
            Various plots for visualization, depending on settings
    """
    plt.style.use('seaborn-darkgrid')

    SJF_labels = ['M/M/1', 'M/M/2', 'M/M/4']
    SJF_colours = ['crimson', 'deepskyblue', 'mediumseagreen']

    labels = [r'$\rho = 0.4$', r'$\rho = 0.77$', r'$\rho = 0.88$', r'$\rho = 0.95$']
    colours = ['yellowgreen', 'goldenrod', 'orange', 'firebrick']

    fig, ax = plt.subplots(1,1,figsize=(12,6), dpi=300)

    rhos = np.linspace(rho_lo, rho_hi, num_rho)

    if FIFO_b == False:
        # FIFO Data for comparison
        file = 'MEAN_WAITS_rho_' + str(num_rho) + '_NSIM' + str(num_reps) + '_' + str(True)+str(Det_b) + str(LongT_b)
        mean_wait = np.load('Data/waitingtimerho/'+file+'.npy')

        # mean over the simulations
        simulation_means = np.mean(mean_wait, axis=2)
        simulation_SDs = np.std(mean_wait, axis=2)

        for r_i in range(3): # r_i = server num
            ax.plot(rhos, simulation_means[r_i,:], color=SJF_colours[r_i], label=rf'({SJF_labels[r_i]}) FIFO', alpha=0.5)
            ax.plot(rhos, simulation_means[r_i,:],'.', color=SJF_colours[r_i], alpha=0.5)

            upper = simulation_means[r_i,:]-2*simulation_SDs[r_i,:]
            lower = simulation_means[r_i,:]+2*simulation_SDs[r_i,:]
            ax.fill_between(rhos, upper, lower, color=SJF_colours[r_i], alpha=0.1)

        # SJF Data
        file = 'MEAN_WAITS_rho_' + str(num_rho) + '_NSIM' + str(num_reps) + '_' +str(FIFO_b)+str(Det_b) + str(LongT_b)
        SJF_mean_wait = np.load('Data/waitingtimerho/'+file+'.npy')

        # mean over the simulations
        SJF_sim_means = np.mean(SJF_mean_wait, axis=2) # prev dim = [n_serv,rho,sim_num] || new dim = [n_serv,rho]
        SJF_sim_SDs = np.std(SJF_mean_wait, axis=2)
        ax.plot(rhos, SJF_sim_means[0,:], color='purple', label=rf'(M/M/1) SJF')
        ax.plot(rhos, SJF_sim_means[0,:], 'x', color='purple')

        upper = SJF_sim_means[0,:] + SJF_sim_SDs[0,:]
        lower = SJF_sim_means[0,:] - SJF_sim_SDs[0,:]
        ax.fill_between(rhos, lower, upper, color='purple', alpha=0.35)
        ax.set_title(rf'FIFO vs SJF | {num_sims} Simulations', fontsize=16)

        
        ax.legend(fancybox=True, shadow=True, fontsize=14, loc='upper left')
        ax.set_xlabel(r'System Load', fontsize=16)
        ax.grid()
    
    else:
        #FIFO Data (Deterministic, Exponential and Hyperexponential service times)
        file = 'MEAN_WAITS_rho_' + str(num_rho) + '_NSIM' + str(num_reps) + '_' + str(FIFO_b)+str(Det_b) + str(LongT_b)
        mean_wait = np.load('Data/waitingtimerho/'+file+'.npy')

        # mean over the simulations
        simulation_means = np.mean(mean_wait, axis=2) # prev dim = [n_serv,rho,sim_num] || new dim = [n_serv,rho]
        simulation_SDs = np.std(mean_wait, axis=2)
        if Det_b:
            ax.set_title(rf'Deterministic Service Times | {num_reps} simulations', fontsize=18)
        else:
            if LongT_b:
                ax.set_title(rf'Hyperexponential Service Times | {num_reps} simulations', fontsize=18)
            else:
                ax.set_title(rf'Exponential Service Times | {num_reps} simulations', fontsize=18)
        
        for i, r_i in enumerate([1, 6, 8, -1]): # i = server number
            ax.plot([1,2,4], simulation_means[:,r_i], color=colours[i], label=rf'({labels[i]})')
            ax.plot([1,2,4], simulation_means[:,r_i],'o', color=colours[i])

            upper = simulation_means[:,r_i]-2*simulation_SDs[:,r_i]
            lower = simulation_means[:,r_i]+2*simulation_SDs[:,r_i]
            ax.fill_between([1,2,4], upper, lower, color=colours[i], alpha=0.2)
        ax.legend(fancybox=True, shadow=True, fontsize=14, loc='upper right')
        ax.set_xlabel(r'Server Number', fontsize=16)
        ax.grid()

    ax.set_ylabel(r'Wait Time', fontsize=16)
    ax.grid()
    fig.tight_layout()
    fig.savefig(f'Figs/Graph_{name}.png')
    plt.show()


######## PLOTTING ALL WAIT TIMES ######################################333

def all_plotter(rho_lo=0.5, rho_hi=0.95, num_reps=50, num_rho=15, save = False, show=False, name="Doolittle"):

    """ Plot various distributions, system loads and server number waiting times
        Args
            rho_lo      minimum server load
            rho_hi      maximum server load
            num_reps    number of measuemrements
            num_rho     number of server load settings
            save        boolean for saving output
        Returns
            Various plots for visualization, depending on settings"""
    plt.style.use('seaborn-darkgrid')

    labels = ['Deterministic', 'Exponential', 'Hyperexponential']
    colours = ['darkviolet', 'mediumblue', 'dodgerblue']

    fig, ax = plt.subplots(1,1,figsize=(12,6), dpi=300)

    # same rho for each queue
    rhos = np.linspace(rho_lo, rho_hi, num_rho)

    for i, type_q in enumerate(['TrueTrueFalse', 'TrueFalseFalse', 'TrueFalseTrue']):
        file_E = 'MEAN_WAITS_rho_' + str(num_rho) + '_NSIM' + str(num_reps) + '_' + type_q
        mean_wait = np.load('Data/waitingtimerho/'+file_E+'.npy')
        # mean over the simulations
        simulation_means = np.mean(mean_wait, axis=2) # prev dim = [n_serv,rho,sim_num] || new dim = [n_serv,rho]
        simulation_SDs = np.std(mean_wait, axis=2)

        ax.set_title('Comparison of Service Time Distributions')

        rho_set = 7 # Change System Load Here
        ax.plot([1,2,4], simulation_means[:,rho_set], color=colours[i], label=rf'({labels[i]})')
        ax.plot([1,2,4], simulation_means[:,rho_set],'o', color=colours[i])

        upper = simulation_means[:,rho_set]-2*simulation_SDs[:,rho_set]
        lower = simulation_means[:,rho_set]+2*simulation_SDs[:,rho_set]
        ax.fill_between([1,2,4], upper, lower, color=colours[i], alpha=0.2)
    

    ax.grid()
    ax.legend(fancybox=True, shadow=True, fontsize=14, loc='upper right')
    ax.set_xlabel(r'Server Number', fontsize=16)
    ax.set_ylabel(r'Wait Time', fontsize=16)

    ax.tick_params(axis='both', labelsize=14)
    ax.grid()
    fig.tight_layout()
    fig.savefig(f'Figs/Graph_{name}.png')
    plt.show()

####### TIMER #################
import time

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))


if __name__ == '__main__':

    """ Run All"""

    ###### Big Run #######
    n_cust = 40000
    r_min = 0.4
    r_max = 0.95
    num_sims = 250
    num_rho = 10

    ###### CODE BELOW CAN BE UNCOMMENTED TO GENERATE NEW DATA

    # start_t = time.time()

    # results = sim(n_cust,rho_lo=r_min, rho_hi=r_max, num_sims=num_sims, num_rho=num_rho,n_serv=True,FIFO_b=True,Det_b=False,LongT_b=False)
    # print('1/4')
    # results_SJF = sim(n_cust,rho_lo=r_min, rho_hi=r_max, num_sims=num_sims, num_rho=num_rho,n_serv=False,FIFO_b=False,Det_b=False,LongT_b=False)
    # print('2/4')
    # results_Deterministic = sim(n_cust,rho_lo=r_min, rho_hi=r_max, num_sims=num_sims, num_rho=num_rho,n_serv=True,FIFO_b=True,Det_b=True,LongT_b=False)
    # print('3/4')
    # results_Longtail = sim(n_cust,rho_lo=r_min, rho_hi=r_max, num_sims=num_sims, num_rho=num_rho,n_serv=True,FIFO_b=True,Det_b=False,LongT_b=True)
    # print('4/4')


    # end_t = time.time()
    # elapsed_t = end_t - start_t
    # time_convert(elapsed_t)

    plotter(rho_lo=r_min, rho_hi=r_max, num_reps=num_sims, num_rho=num_rho, FIFO_b=True, Det_b=False, LongT_b=False, name='FIFO_servers')
    plotter(rho_lo=r_min, rho_hi=r_max, num_reps=num_sims, num_rho=num_rho, FIFO_b=False, Det_b=False, LongT_b=False, name='SJF_vsd_FIFO')
    plotter(rho_lo=r_min, rho_hi=r_max, num_reps=num_sims, num_rho=num_rho,FIFO_b=True, Det_b=True, LongT_b=False, name='Deterministic_servers')
    plotter(rho_lo=r_min, rho_hi=r_max, num_reps=num_sims, num_rho=num_rho,FIFO_b=True, Det_b=False, LongT_b=True, name='Longtail_servers')

    ### COMPARING SERVICE TIME DISTRIBUTIONS ################################3

    all_plotter(rho_lo=r_min, rho_hi=r_max, num_reps=num_sims, num_rho=num_rho, name='Distro_comp')
