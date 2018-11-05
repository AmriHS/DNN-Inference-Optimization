# An example using Keras Resnet50 pre-trained model to measure the inference time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os
from apscheduler.schedulers.background import BackgroundScheduler
#import apscheduler.schedulers.blocking
import commands

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from timeit import default_timer as timer
from keras.datasets import cifar10

import keras.backend.tensorflow_backend as ktf
import tensorflow as tf
import numpy as np
import cv2 #, os
import csv

import subprocess

import logging

#log = logging.getLogger('apscheduler.executors.default')
#log.setLevel(logging.INFO)  # DEBUG

#fmt = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
#h = logging.StreamHandler()
#h.setFormatter(fmt)
#log.addHandler(h)


logger = logging.getLogger()  # this returns the root logger 
logger.addHandler(logging.StreamHandler())
Time = []
PowerConsumption = []


#imgs = []
#for file in os.listdir("C://Users//hcaro//OneDrive//Documents//images"):
#    img = cv2.imread(os.path.join("C://Users//hcaro//OneDrive//Documents//images",file))
#    img = cv2.resize(img, (224, 224))
#    imgs.append(img)
def tick():
#     CurrentPowerConsumption = commands.getoutput('cat /sys/devices/7000c400.i2c/i2c-1/1-0040/iio_device/in_power0_input')
#     CurrentPowerConsumption = 1000
#    p = subprocess.Popen(["cat", "/sys/devices/7000c400.i2c/i2c-1/1-0040/iio_device/in_power0_input"], stdout=subprocess.PIPE)
#    for line in iter(p.stdout.readline, ''):
        # do something with line
#        var=line
#        print line,
#    p.stdout.close()
#    exit_code = p.wait() # wait for the process to exit
#    CurrentPowerConsumption=var
    input = open('/sys/devices/7000c400.i2c/i2c-1/1-0040/iio_device/in_power0_input', 'r')  
    CurrentPowerConsumption=input.readline()   
    print(CurrentPowerConsumption)
    PowerConsumption.append(float(CurrentPowerConsumption))
#    CurrentTime = commands.getoutput('date + %S')
    CurrentTime = time.time()
    Time.append(int(CurrentTime))

    #print('Tick! The time is: %s' % datetime.now())

def RelationPlot(Time,PowerConsumption):
    plt.plot(Time, PowerConsumption)
    plt.xlabel('Time')
    plt.ylabel('PowerConsumption')
    plt.savefig("PowerConsumption_test_timer.jpg")
    print("Average PowerConsumption:")
    Power_Sum =0
    for i in range(len(PowerConsumption)):
        Power_Sum+=PowerConsumption[i]
    print(Power_Sum/len(PowerConsumption))
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
        #x = np.expand_dims(x, axis=0)
        processed_data.append(preprocess_input(x))
    return processed_data


def make_predictions(dataset, batch_size, allow_growth, memory_frac):
    print('getting into models!')
    scheduler = BackgroundScheduler()
#    scheduler = apscheduler.schedulers.blocking.BackgroundScheduler('apscheduler.job_defaults.max_instances': '2')
    print('BackgroundScheduler define')
    scheduler.add_job(tick, 'interval', seconds=0.5, misfire_grace_time=1000)# execute every 0.5 second
    print('job added!')

    config = tf.ConfigProto(log_device_placement=False, device_count = {'GPU' :1})
    if allow_growth:
	config.gpu_options.allow_growth = True
    else:
	config.gpu_options.per_process_gpu_memory_fraction = memory_frac

    print ("Memory Fraction:"+repr(memory_frac))
    print ("Allow Growth:"+repr(allow_growth))
    print ("Batch Size:"+repr(batch_size))

    session = tf.Session(config=config)
    ktf.set_session(session)

    model = ResNet50(weights='imagenet')
    # start time
    try:
    	scheduler.start()# new seperate thread
        print('Press Ctrl+{0} to exit'.format('Break' if os.name == 'nt' else 'C'))
        start = timer()
        preds = model.predict(np.array(dataset), batch_size=batch_size)
    	# end time
    	end = timer()
    except (KeyboardInterrupt, SystemExit):
           # Not strictly necessary if daemonic mode is enabled but should be done if possible
           scheduler.shutdown()
    scheduler.shutdown()
    print('Exit The Job!')
    RelationPlot(Time,PowerConsumption)
    # calculate runtime
    runtime = end-start
    print('Runtime: ' + "{0:.2f}".format(runtime) + 's')
    return preds, runtime

def run_resnet50_benchmark(batch_size, all_growth=True, mem_frac=None):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print (len(X_train))
    dataset = resize(X_train[:100])
    preds, runtime = make_predictions(dataset, batch_size, all_growth, mem_frac)
    decoded =  decode_predictions(preds, top=1)
    write_to_csv(decoded, "Keras_result.csv")
    return 'Resnet50', runtime
