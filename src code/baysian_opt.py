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

# set seed for random value
random.seed(24)

# host IP and credential
ip='10.173.131.120'
port=22
username='ubuntu'
password='ubuntu'

#Linux commands
file_dir = 'cd DNN_Inference;'
sudo_cmd = 'sudo -S '
command='python run_benchmark.py --cpu_freq {0} --num_cores {1} --gpu_freq {2} --emc_freq {3} --bsize {4}' \
    +' --all_growth {5} --mem_frac {6}'

# define discrete values of each input space
batch_size_array=[1, 8, 16, 32] # 16 & 32 is excluded for VGG16
all_growth_array=[True, False]
memory_frac_array=[0.15, 0.2, 0.25, 0.3, 0.33] # 0.15, 0.2,  is excluded for VGG16 model
GPU_freq_array=[76800000,153600000, 230400000, 307200000, 384000000, 460800000, 537600000, 614400000,
                691200000, 768000000, 844800000, 921600000, 998400000, 537600000,998400000]
CPU_freq_array=[102000,204000,306000, 408000, 510000, 612000, 714000, 816000, 918000,1020000,1122000, 1224000,
                1326000,1224000,1428000,1555000,1632000, 1734000]

# 40800000 frequency excluded due to frequent memory issue
EMC_freq_array=[800000000,1065600000,1331200000,1600000000]
num_dis_core_array=[0,1,2,3]

# define space input, upper and lower bounds for each dimension
domain = gpflowopt.domain.ContinuousParameter('CPU_frequency', 102000, 1734000) + \
         gpflowopt.domain.ContinuousParameter('num_cores_dis', 0, 4) + \
         gpflowopt.domain.ContinuousParameter('GPU_frequency', 76800000, 998400000) + \
         gpflowopt.domain.ContinuousParameter('EMC_frequency', 800000000, 1600000000)+ \
         gpflowopt.domain.ContinuousParameter('Batch_size', 1, 32) + \
         gpflowopt.domain.ContinuousParameter('Allow_growth', 0, 1) + \
         gpflowopt.domain.ContinuousParameter('Memory_fraction', 0.15, 0.33)

# connect to linux server using ssh
ssh=paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(ip,port,username,password)

# write output to csv file
def write_to_csv(data, filename):
    with open(filename,'a') as out:
        csv_out= csv.writer(out, lineterminator='\n')
        for row in data:
            csv_out.writerow(row)

# find closest discrete values to the continuous input
def find_closest(values):
    values[0] = min(CPU_freq_array, key=lambda x:abs(x-values[0]))
    values[1] = min(num_dis_core_array, key=lambda x:abs(x-values[1]))
    values[2] = min(GPU_freq_array, key=lambda x:abs(x-values[2]))
    values[3] = min(EMC_freq_array, key=lambda x:abs(x-values[3]))
    values[4] = min(batch_size_array, key=lambda x:abs(x-values[4]))
    values[5] = min(all_growth_array, key=lambda x:abs(x-values[5]))
    values[6] = min(memory_frac_array, key=lambda x:abs(x-values[6]))


def handle_output(output):
    if len(output) == 0:
        return [None, None, None, None]
    else:
        return [float(output[0]), float(output[1]), float(output[2]), float(output[3])]

# Optimization multo-objective function
def objective_func(params):
    y1 = []
    y2 = []
    benchmark_data = []
    for i in range(len(params)):
        # find closest discrete value
        find_closest(params[i])

        # formatting linux command
        cmd = command.format(int(params[i,0]), int(params[i,1]),int(params[i,2]),int(params[i,3]),int(params[i,4]),
                             int(params[i,5]),float(params[i,6]))
        print ("Command:"+repr(cmd))

        # SSH requests
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

    # return objective values
    return np.hstack((y1,y2))

def random_search():
    n_samples = 40
    design = gpflowopt.design.RandomDesign(n_samples, domain)
    X = design.generate()
    Y = objective_func(X)
    itemindex = np.where(Y[:n_samples]==None) #discard samples where it doesn't produce output
    Y = np.delete(Y, (itemindex[0]), axis=0)
    Y = np.array(Y, dtype=float)

    plt.scatter(Y[:,0], Y[:,1])
    plt.title('Random set')
    plt.xlabel('Inference Time')
    plt.ylabel('Power Consumption')
    plt.show()

    plt.plot(np.arange(0, Y.shape[0]),np.minimum.accumulate(Y[:,0]) ,'b',label='Inference Time')
    plt.ylabel('fmin')
    plt.xlabel('Number of evaluated points')
    plt.legend()
    plt.show()

    plt.plot(np.arange(0, Y.shape[0]),np.minimum.accumulate(Y[:,1]) ,'g',label='Power Consumption')
    plt.ylabel('fmin')
    plt.xlabel('Number of evaluated points')
    plt.legend()
    plt.show()

def baysian_opt():
    global is_sampling
    n_samples = 10
    design = gpflowopt.design.LatinHyperCube(n_samples, domain)
    X = design.generate()
    X = np.array(X)
    Y = objective_func(X)
    # discard samples where it doesn't produce output
    itemindex = np.where(Y[:n_samples]==None)
    Y = np.delete(Y, (itemindex[0]), axis=0)
    X = np.delete(X, (itemindex[0]), axis=0)
    Y = np.array(Y, dtype=float)
    n_samples = len(X)

    # One model for each objective
    objective_models = [gpflow.gpr.GPR(X.copy(), Y[:,[i]].copy(), gpflow.kernels.Matern52(domain.size, ARD=True)) for i in range(Y.shape[1])]
    for model in objective_models:
        model.likelihood.variance = 0.01

    hvpoi = gpflowopt.acquisition.HVProbabilityOfImprovement(objective_models)
    acquisition_opt = gpflowopt.optim.StagedOptimizer([gpflowopt.optim.MCOptimizer(domain, n_samples),
                                                       gpflowopt.optim.SciPyOptimizer(domain)])

    # Then run the BayesianOptimizer for 40 iterations
    optimizer = gpflowopt.BayesianOptimizer(domain, hvpoi, optimizer=acquisition_opt, verbose=True)
    optimizer.optimize(objective_func, n_iter=30)

    pf, dom = gpflowopt.pareto.non_dominated_sort(hvpoi.data[1])

    plt.scatter(hvpoi.data[1][:,0], hvpoi.data[1][:,1], c=dom)
    plt.title('Pareto set')
    plt.xlabel('Inference Time')
    plt.ylabel('Power Consumption')
    plt.show()

    plt.plot(np.arange(0, hvpoi.data[0].shape[0]),np.minimum.accumulate(hvpoi.data[1][:,0]) ,'b',label='Inference Time')
    plt.ylabel('fmin')
    plt.xlabel('Number of evaluated points')
    plt.legend()
    plt.show()

    plt.plot(np.arange(0, hvpoi.data[0].shape[0]),np.minimum.accumulate(hvpoi.data[1][:,1]) ,'g',label='Power Consumption')
    plt.ylabel('fmin')
    plt.xlabel('Number of evaluated points')
    plt.legend()
    plt.show()

# run Bayesian Optimization and Random Design
baysian_opt()
random_search()