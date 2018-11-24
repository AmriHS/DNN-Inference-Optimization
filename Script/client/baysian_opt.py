import paramiko
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import gpflow
import gpflowopt
import numpy as np
import random
import time
import csv
from gpflowopt.acquisition import ExpectedImprovement
from random import randint
random.seed(24)

# host IP and credential
ip='10.173.131.120'
port=22
username='ubuntu'
password='ubuntu'

#Linux commands
file_dir = 'cd DNN_Inference;'
sudo_cmd = 'sudo -S '
command='python run_bench_v2.py --cpu_freq {0} --num_cores {1} --gpu_freq {2} --emc_freq {3} --bsize {4}' \
    +' --all_growth {5} --mem_frac {6}'
command2='python DNN_Inference/ssh_ex.py'

# define discrete values of each input space
batch_size_array=[1, 8, 16, 32]
all_growth_array=[True, False]
memory_frac_array=[0.1, 0.15, 0.2, 0.25, 0.3, 0.33] # 0.1 is exluded due to memory issue
GPU_freq_array=[76800000,153600000, 230400000, 307200000, 384000000, 460800000, 537600000, 614400000,
                691200000, 768000000, 844800000, 921600000, 998400000, 537600000,998400000]
CPU_freq_array=[102000,204000,306000, 408000, 510000, 612000, 714000, 816000, 918000,1020000,1122000, 1224000,
                1326000,1224000,1428000,1555000,1632000, 1734000]
EMC_freq_array=[800000000,1065600000,1331200000,1600000000]
# 40800000 frequency excluded due to memory issue
num_dis_core_array=[0,1,2,3]

# connect to linux server using ssh
ssh=paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(ip,port,username,password)

# write output to csv file
def write_to_csv(data, filename):
    with open(filename,'a') as out:
        csv_out= csv.writer(out, lineterminator='\n')
        #csv_out.writerow(['CPU Frequency', '# of enabled cores','GPU Frequency','EMC Frequency','Batch Size', 'Mem Growth',
        #                  'Mem Fraction', 'Runtime', 'Model Cons','GPU Cons', 'CPU Cons'])
        for row in data:
            csv_out.writerow(row)

def vlmop2(x):
    transl = 1 / np.sqrt(2)
    part1 = (x[:, [0]] - transl) ** 2 + (x[:, [1]] - transl) ** 2
    part2 = (x[:, [0]] + transl) ** 2 + (x[:, [1]] + transl) ** 2
    y1 = 1 - np.exp(-1 * part1)
    y2 = 1 - np.exp(-1 * part2)
    return np.hstack((y1, y2))

# round input values
def round(params):
    params[:, [0]] = np.round(params[:, [0]]*1000000, decimals=0)
    params[:, [1]] = np.round(params[:, [1]], decimals=0)
    params[:, [2]] = np.round(params[:, [2]]*1000000, decimals=0)
    params[:, [3]] = np.round(params[:, [3]]*1000000000, decimals=0)
    params[:, [4]] = np.round(params[:, [4]], decimals=2)
    params[:, [6]] = np.round(params[:, [6]], decimals=2)

# find closest discrete values to the continuous input
def find_closest(values):
    values[0] = min(CPU_freq_array, key=lambda x:abs(x-values[0]))
    values[1] = min(num_dis_core_array, key=lambda x:abs(x-values[1]))
    values[2] = min(GPU_freq_array, key=lambda x:abs(x-values[2]))
    values[3] = min(EMC_freq_array, key=lambda x:abs(x-values[3]))
    values[4] = min(batch_size_array, key=lambda x:abs(x-values[4]))
    values[5] = min(all_growth_array, key=lambda x:abs(x-values[5]))
    values[6] = min(memory_frac_array, key=lambda x:abs(x-values[6]))

is_sampling = True

def handle_output(output):
    if len(output) == 0:
        output = [None, None,None, None]
    elif not is_sampling and len(output) != 0:
        output = [float(output[0]), float(output[1]), float(output[2]), float(output[3])]
    return output

# Optimization multo-objective function
def objective_func(params):
    print (params.shape)
    y1 = []
    y2 = []
    benchmark_data = []
    for i in range(len(params)):
        print ("Iteration:"+str(i))
        find_closest(params[i])
        cmd = command.format(int(params[i,0]), int(params[i,1]),int(params[i,2]),int(params[i,3]),int(params[i,4]),
                             int(params[i,5]),float(params[i,6]))
        print ("Command:"+repr(cmd))
        stdin,stdout,stderr=ssh.exec_command(file_dir+sudo_cmd+cmd)
        stdin.write("ubuntu\n")
        stdin.flush()
        outlines=stdout.readlines()
        response=''.join(outlines).split()
        response = handle_output(response)
        benchmark_data.append([params[i,0], params[i,1], params[i,2], params[i,3], params[i,4], params[i,5],
                         params[i,6], response[0], response[1], response[2], response[3]])
        y1.append([response[0]])
        y2.append([response[1]])
    write_to_csv(benchmark_data, "result.csv")
    return np.hstack((y1,y2))

def random_search(n_iter):
    params = []
    for i in range(1,n_iter+1):
        cpu_r = random.randint(102000, 1734000)
        core_r = random.randint(0, 4)
        gpu_r = random.randint(76800000, 998400000)
        emc_r = random.randint(800000000, 1600000000)
        batch_r = random.randint(1, 32)
        mem_r = random.randint(0,1)
        grow_r = random.randint(0.15, 0.33)
        params.append([cpu_r, core_r, gpu_r, emc_r, batch_r, mem_r, grow_r])
    return objective_func(params)

def baysian_opt():
    # define space input, upper and lower bounds for each dimension
    domain = gpflowopt.domain.ContinuousParameter('CPU_frequency', 102000, 1734000) + \
             gpflowopt.domain.ContinuousParameter('num_cores_dis', 0, 4) + \
             gpflowopt.domain.ContinuousParameter('GPU_frequency', 76800000, 998400000) + \
             gpflowopt.domain.ContinuousParameter('EMC_frequency', 800000000, 1600000000)+ \
             gpflowopt.domain.ContinuousParameter('Batch_size', 1, 32) + \
             gpflowopt.domain.ContinuousParameter('Allow_growth', 0, 1) + \
             gpflowopt.domain.ContinuousParameter('Memory_fraction', 0.15, 0.33)
# Setup input domain
    #domain = gpflowopt.domain.ContinuousParameter('x1', -2, 2) + \
    #     gpflowopt.domain.ContinuousParameter('x2', -2, 2)
    # EMC frequency -> 0.0408 discarded due to memory issues
    n_samples = 2
    design = gpflowopt.design.RandomDesign(n_samples, domain, )
    X = design.generate()
    Y = objective_func(X)
    itemindex = np.where(Y[:n_samples]==None) #discard samples where it doesn't produce output
    Y = np.delete(Y, (itemindex[0]), axis=0)
    X = np.delete(X, (itemindex[0]), axis=0)
    Y = np.array(Y, dtype=float)
    n_samples = len(X)
    is_sampling = False
    print (X)
    print (Y)
    # One model for each objective
    objective_models = [gpflow.gpr.GPR(X.copy(), Y[:,[i]].copy(), gpflow.kernels.Matern52(domain.size, ARD=True)) for i in range(Y.shape[1])]
    for model in objective_models:
        model.likelihood.variance = 0.01

    hvpoi = gpflowopt.acquisition.HVProbabilityOfImprovement(objective_models)
    acquisition_opt = gpflowopt.optim.StagedOptimizer([gpflowopt.optim.MCOptimizer(domain, n_samples),
                                                       gpflowopt.optim.SciPyOptimizer(domain)])

# Then run the BayesianOptimizer for 50 iterations
    optimizer = gpflowopt.BayesianOptimizer(domain, hvpoi, optimizer=acquisition_opt, verbose=True)
    result = optimizer.optimize(objective_func, n_iter=20)

    #print(result)

    pf, dom = gpflowopt.pareto.non_dominated_sort(hvpoi.data[1])
    print ("::::::::::::::::::::::::::::::::::::::::::::::::::")
    print (hvpoi.data[1][:,0])
    print ("##########################")
    print (hvpoi.data[1][:,1])

    plt.scatter(hvpoi.data[1][:,0], hvpoi.data[1][:,1], c=dom)
    plt.title('Pareto set')
    plt.xlabel('Inference Time')
    plt.ylabel('Power Consumption')
    plt.show()

    plt.plot(np.arange(0, hvpoi.data[0].shape[0]),np.minimum.accumulate(hvpoi.data[1][:,0]) ,'b',label='Inference Time')
    plt.plot(np.arange(0, hvpoi.data[0].shape[0]),np.minimum.accumulate(hvpoi.data[1][:,1]) ,'g',label='Power Consumption')
    plt.ylabel('fmin')
    plt.xlabel('Number of evaluated points')
    plt.legend()
    plt.show()
baysian_opt()