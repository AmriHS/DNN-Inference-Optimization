""" Main entry point for running benchmarks with different Keras backends."""

import tensorflow as tf
import argparse
import keras
import json
import csv
from models import Keras_Resnet50 as res50

def write_to_csv(data, filename):
    with open(filename,'w') as out:
        csv_out= csv.writer(out, lineterminator='\n')
        csv_out.writerow(['Model', 'Runtime'])
        for row in data:
            csv_out.writerow(row)


#parser = argparse.ArgumentParser()
#parser.add_argument('--config',
                    #help='There are multiple configurations range from 1 to 3, choose one.')

#args = parser.parse_args()

# Load the json config file for the requested mode.
#config_file = open("benchmarks/scripts/keras_benchmarks/config.json", 'r')
#config_contents = config_file.read()
#config = json.loads(config_contents)[args.config]

#Benchmark runtime data
benchmark_data = []

#run benchmark for Keras Resnet50
resnet_benchmark = res50.run_resnet50_benchmark()
benchmark_data.append(resnet_benchmark)

#Write banchmarks result
write_to_csv(benchmark_data, "result.csv")