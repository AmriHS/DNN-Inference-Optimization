import tensorflow as tf
import argparse
import keras
import json
import os
import cv2
import Keras_Resnet50 as res50
import subprocess
import time
import signal

def load_cifar10_images():
    imgs = []
    for file in os.listdir("/home/ubuntu/DNN_Inference/cifar/test"):
        img = cv2.imread(os.path.join("/home/ubuntu/DNN_Inference/cifar/test",file))
        img = cv2.resize(img, (224, 224))
        imgs.append(img)
    return imgs

def run_script(cpu_freq, num_dis_core, GPU_freq, EMC_freq):
    command= "bash ./script.sh "+str(cpu_freq)+" "+str(num_dis_core)+" "+str(GPU_freq)+" "+str(EMC_freq)
    p = subprocess.Popen(command, shell=True)
    time.sleep(10)
    os.kill(p.pid, signal.SIGINT)

#Default values 
all_growth = True
batch_size = 1
mem_frac_per_GPU = None

parser = argparse.ArgumentParser()
parser.add_argument('--cpu_freq', help='Allow memory growth for GPU during the runtime')
parser.add_argument('--num_cores', help='Allow memory growth for GPU during the runtime')
parser.add_argument('--gpu_freq', help='Allow memory growth for GPU during the runtime')
parser.add_argument('--emc_freq', help='Allow memory growth for GPU during the runtime')
parser.add_argument('--all_growth', help='Allow memory growth for GPU during the runtime')
parser.add_argument('--mem_frac', help='Set a fixed value for memory fraction per GPU')
parser.add_argument('--bsize', help='Set sample batch size for prediction')
args = parser.parse_args()

#if args.all_growth == 1 and args.mem_frac > 0:
#	raise ValueError('Either all_growth or mem_frac args should be specificed. Not both')

cpu_freq = int(args.cpu_freq)
num_cores = int(args.num_cores)
gpu_freq = int(args.gpu_freq)
emc_freq = int(args.emc_freq)
batch_size = int(args.bsize)

if args.mem_frac: 
	mem_frac_per_GPU = float(args.mem_frac)
	all_growth = False
elif args.all_growth:
	all_growth = True
#Benchmark runtime data
benchmark_data = []

# for maximum frequency
run_script(1734000, 0, 998400000, 1600000000)


dataset = load_cifar10_images()
run_script(cpu_freq, num_cores, gpu_freq, emc_freq)

#run benchmark for Keras Resnet50
runtime, pow_cons = res50.run_resnet50_benchmark(dataset, batch_size, all_growth, mem_frac_per_GPU)

print(runtime)
print(pow_cons[0])
print(pow_cons[1])
print(pow_cons[2])
print(pow_cons[3])