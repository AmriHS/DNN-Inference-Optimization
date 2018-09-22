# An example using Keras Resnet50 pre-trained model to measure the inference time
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


#imgs = []
#for file in os.listdir("C://Users//hcaro//OneDrive//Documents//images"):
#    img = cv2.imread(os.path.join("C://Users//hcaro//OneDrive//Documents//images",file))
#    img = cv2.resize(img, (224, 224))
#    imgs.append(img)

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


def make_predictions(dataset, num_cores, threads_parallel, batch_size):
    config = tf.ConfigProto(intra_op_parallelism_threads=8,
                        inter_op_parallelism_threads=8, allow_soft_placement=True, log_device_placement=False,
                        device_count = {'CPU' : num_cores, 'GPU' : 0}) #cpu_memory=3.75
    session = tf.Session(config=config)
    ktf.set_session(session)

    model = ResNet50(weights='imagenet')
    # start time
    start = start = timer()
    preds = model.predict(np.array(dataset), batch_size=batch_size)
    # end time
    end = timer()
    # calculate runtime
    elapsed_time_batch = end-start
    print('Runtime: ' + "{0:.2f}".format(end-start) + 's')
    return preds

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
dataset = resize(X_train[:200])
preds = make_predictions(dataset, num_cores=4, threads_parallel=16, batch_size=32)
decoded =  decode_predictions(preds, top=1)
write_to_csv(decoded, "result1.csv")
