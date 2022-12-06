import scipy.stats as stat
from scipy.stats import norm
import numpy as np

############## STATISTICAL TEST FUNCTIONS ###################################################

# Compare variance of 2 samples
def f_test(data1, data2, conf_level=0.99, SJF_b=False):
    """
    Compares variances of 2 datasets
    Args
        data1       first dataset
        data2       second dataset
        conf_level  confidence level
        SJF_b       boolean for Shortest Job First data
    Returns
        F statistic
        P value
        Boolean of variances equal / unequal
    """
    f = np.var(data1, ddof=1)/np.var(data2,ddof=1)
    nun = data1.size - 1
    dun = data2.size - 1
    p_val = 1- stat.f.cdf(f,nun,dun)

    if not SJF_b:
        if p_val <= (1-conf_level):
            print('unequal variances')
            same = False
        else:
            print('equal variances')
            same = True
    else:
        same = True

    return f, p_val, same

group1 = np.array([0.28, 0.2, 0.26, 0.28, 0.5])
group2 = np.array([444, 23, 544, 500, 5000])

# f_out = f_test(group1, group2)

def welch(data1, data2, cl = 0.95, SJF_b=False):
    """
    Checks whether means of 2 datasets are significantly different
    Args
        data1       first dataset
        data2       second dataset
        cl          confidence level
        SJF_b       boolean for Shortest Job First data
    Returns
        t statistic
        confidence level
    """
    var_bool = f_test(data1,data2,conf_level=0.95, SJF_b=SJF_b)[2]

    
    w = stat.ttest_ind(data1, data2, equal_var=var_bool)
    w = w[1]

    return w, cl

################# SIGNIFICANCE TESTING #############################################


# Data shape: [n_serv, rho, sim_num] = [3, 10, 150] (Has already been averaged over customers)
fifo_data = np.load('Data/waitingtimerho/MEAN_WAITS_rho_10_NSIM250_TrueFalseFalse.npy')
SJF_data = np.load('Data/waitingtimerho/MEAN_WAITS_rho_10_NSIM250_FalseFalseFalse.npy')
Det_data = np.load('Data/waitingtimerho/MEAN_WAITS_rho_10_NSIM250_TrueTrueFalse.npy')
Longtail_data = np.load('Data/waitingtimerho/MEAN_WAITS_rho_10_NSIM250_TrueFalseTrue.npy')


######### DISTRIBUTION COMPARISON RUNS ###################################################
def performance_comp(data1, data2, n1, n2, rho_index, SJF_b=False):
    """
    Compares performance of M/M/1 server with M/M/2 and M/M/4  by testing for a significant
    difference in waiting times.
    Args
        data: Data from given simulation (FIFO, SJF, DeterministicFIFO, LongTail)
        n1:   First server index | (0, 1, 2) -> (1, 2, 4)
        n2:   Second server index
        rho:  Index for rho (Server load) to be investigated
    Returns
        Welch Test Output
        Descriptions of results
    
    """
    server_nums = [1,2,4]
    rhos = (np.linspace(0.4, 0.95, 10))


    n_a = data1[n1, rho_index, :] # 150 measurements generated at first server number
    n_b = data2[n2, rho_index, :] # 150 measurements generated at second server number

    # Means, SDs and Confidence Intervals
    mean_a = np.mean(n_a)
    std_a = np.std(n_a)
    CI_l, CI_h = norm.interval(confidence=0.95, loc=mean_a, scale=std_a)
    CI_a = CI_h - CI_l

    mean_b = np.mean(n_b)
    std_b = np.std(n_b)
    CI_l, CI_h = norm.interval(confidence=0.95, loc=mean_b, scale=std_b)
    CI_b = CI_h - CI_l

    print(f'''Mean for {server_nums[n1]} Servers: {round(mean_a,3)}s. 95% CI: {round(CI_a,3)}s''') 
    if not SJF_b:
        print(f'''Mean for {server_nums[n2]} Servers: {round(mean_b,3)}s. 95% CI: {round(CI_b,3)}s''') 
    print(f'''Server load: {rhos[rho_index]}''')
    
    w, cl = welch(n_a, n_b, SJF_b=SJF_b)

    if not SJF_b:
        if w <= (1-cl):
            print(f'Mean waiting time significantly different: p={w}')
        else:
            print(f'Mean waiting time not significantly different: p={w}')




############# RESULTS ########################################
self_tups = [(0,1), (0,2), (1,2)]
tups_labels = [(1,2), (1,4), (2, 4)]

rho_index = -1
labels = ['FIFO_data', 'Deterministic_data', 'Longtail_data']

for i, type_d in enumerate([fifo_data, Det_data, Longtail_data]):
    print(f'{labels[i]}: RESULTS ############################################################ \n')
    for n in range(3):
        print(f'n{tups_labels[n][0]} vs n{tups_labels[n][1]}')
        self_comp = performance_comp(type_d, type_d, self_tups[n][0], self_tups[n][1], rho_index)
        print('\n\n\n')

print('SJF RESULTS ##########################################################################')
SJF_res = performance_comp(SJF_data, SJF_data, 0, 0, rho_index, SJF_b=True)

