# import packages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os
import commands
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf
import numpy as np
import cv2
import csv
import gc
import subprocess
import logging

from apscheduler.schedulers.background import BackgroundScheduler
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from timeit import default_timer as timer
from keras.datasets import cifar10
from multiprocessing import Process, Queue




os.environ["CUDA_VISIBLE_DEVICES"]="0"
logger = logging.getLogger()  # this returns the root logger
logger.addHandler(logging.StreamHandler())

# global variable to store power consumption consumed by various threads
power_cons = []

# Read power consumption by a separate thread every 0.5 second
def tick(): 
    # Read current power consumptions
    input0 = open('/sys/devices/7000c400.i2c/i2c-1/1-0040/iio_device/in_power0_input', 'r')
    input1 = open('/sys/devices/7000c400.i2c/i2c-1/1-0040/iio_device/in_power1_input', 'r')
    input2 = open('/sys/devices/7000c400.i2c/i2c-1/1-0040/iio_device/in_power2_input', 'r')
    mod_power = input0.readline()
    gpu_power = input1.readline()
    cpu_power = input2.readline()
    power_cons.append([float(mod_power), float(gpu_power), float(cpu_power)])

def calculate_power():
    len_power=len(power_cons)
    mod_power_sum = 0
    gpu_power_sum = 0
    cpu_power_sum = 0
    for i in range(len_power):
        mod_power_sum+=power_cons[i][0]
        gpu_power_sum+=power_cons[i][1]
        cpu_power_sum+=power_cons[i][2]
    mod_power_sum/=len_power
    gpu_power_sum/=len_power
    cpu_power_sum/=len_power
    return [mod_power_sum, gpu_power_sum, cpu_power_sum]

def write_to_csv(data, filename):
    with open(filename,'w') as out:
        csv_out= csv.writer(out, lineterminator='\n')
        csv_out.writerow(['Class', 'Prob'])
        for row in data:
            csv_out.writerow(row[0][1:])

def resize(dataset):
    processed_data = []
    for i in range(len(dataset)):
        x = cv2.resize(dataset[i], (224,224))
        x = image.img_to_array(x)
        processed_data.append(preprocess_input(x))
    return processed_data


def make_predictions(dataset, batch_size, allow_growth, memory_frac):
    scheduler = BackgroundScheduler()
    scheduler.add_job(tick, 'interval', seconds=0.5, misfire_grace_time=1) #execute every 0.5 second

    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU' :1})
    if allow_growth:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = memory_frac

    session = tf.Session(config=config)
    ktf.set_session(session)

    model = ResNet50(weights='imagenet')
    try:
        scheduler.start() # new thread
        start = timer()
        preds = model.predict(np.array(dataset), batch_size=batch_size)
        end = timer()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()

    scheduler.shutdown()
    session.close()
    del session
    #force Garbage collector to remove session
    gc.collect()
    power_cons = calculate_power()

    # calculate runtime
    runtime = end-start
    return preds, runtime, power_cons

def run_resnet50_benchmark(dataset, batch_size, all_growth=True, mem_frac=None):
    preds, runtime, power_cons = make_predictions(dataset[:100], batch_size, all_growth, mem_frac)
    decoded =  decode_predictions(preds, top=1)
    write_to_csv(decoded, "Keras_result.csv")
    return runtime, power_cons



