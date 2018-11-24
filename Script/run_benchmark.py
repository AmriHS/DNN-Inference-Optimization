""" Main entry point for running benchmarks with different Keras backends."""
import tensorflow as tf
import argparse
import keras
import json
import csv
import subprocess
import Keras_Resnet50 as res50
def write_to_csv(data, filename):
    with open(filename,'w') as out:
        csv_out= csv.writer(out, lineterminator='\n')
        csv_out.writerow(['Model', 'Runtime'])
        for row in data:
            csv_out.writerow(row)

#Default values 
all_growth = True
batch_size = 1
mem_frac_per_GPU = None

batch_size_array=[1,8,16,32,64]
ALL_MEM_GROWTH_ARRAY=[0.1, 0.15, 0.2, 0.25, 0.3, 0.33, 0.4]
GPU_FREQ_ARRAY=[76800000,537600000,998400000)
CPU_FREQ_ARRAY=[102000,,918000,1734000)
EMC_FREQ_ARRAY=[40800000,800000000,1000000000,1600000000)
DISABLE_CORE_ARRAY=[0,1,2,3]

#parser = argparse.ArgumentParser()
#parser.add_argument('--all_growth', help='Allow memory growth for GPU during the runtime')
#parser.add_argument('--mem_frac', help='Set a fixed value for memory fraction per GPU')
#parser.add_argument('--bsize', help='Set sample batch size for prediction')
#args = parser.parse_args()

print ("start")
command = "./sleep.sh "+str(CPU_FREQ_ARRAY[2])+" "+str(DISABLE_CORE_ARRAY[0])+" "+str(GPU_FREQ_ARRAY[2])+" "+str(EMC_FREQ_ARRAY[2])
subprocess.call(command, shell=True)
print ("end")

#if args.all_growth == 1 and args.mem_frac > 0:
#	raise ValueError('Either all_growth or mem_frac args should be specificed. Not both')

#if args.bsize:
#	batch_size = int(args.bsize)

#if args.mem_frac: 
#	print(args.mem_frac)
#	mem_frac_per_GPU = float(args.mem_frac)
#	all_growth = False
#elif args.all_growth:
#	all_growth = True
#Benchmark runtime data
#benchmark_data = []

#run benchmark for Keras Resnet50
resnet_benchmark = res50.run_resnet50_benchmark(batch_size, all_growth, mem_frac_per_GPU)
benchmark_data.append(resnet_benchmark)

#Write banchmarks result
write_to_csv(benchmark_data, "result.csv")
