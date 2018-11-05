""" Main entry point for running benchmarks with different Keras backends."""
import tensorflow as tf
import argparse
import keras
import json
import csv
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

parser = argparse.ArgumentParser()
parser.add_argument('--all_growth', help='Allow memory growth for GPU during the runtime')
parser.add_argument('--mem_frac', help='Set a fixed value for memory fraction per GPU')
parser.add_argument('--bsize', help='Set sample batch size for prediction')
args = parser.parse_args()

if args.all_growth == 1 and args.mem_frac > 0:
	raise ValueError('Either all_growth or mem_frac args should be specificed. Not both')

if args.bsize:
	batch_size = int(args.bsize)

if args.mem_frac: 
	print(args.mem_frac)
	mem_frac_per_GPU = float(args.mem_frac)
	all_growth = False
elif args.all_growth:
	all_growth = True
#Benchmark runtime data
benchmark_data = []

#run benchmark for Keras Resnet50
resnet_benchmark = res50.run_resnet50_benchmark(batch_size, all_growth, mem_frac_per_GPU)
benchmark_data.append(resnet_benchmark)

#Write banchmarks result
write_to_csv(benchmark_data, "result.csv")
