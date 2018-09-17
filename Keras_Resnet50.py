# An example using Keras Resnet50 pre-trained model to measure the inference time

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import keras.backend.tensorflow_backend as ktf
import tensorflow as tf
import numpy as np
from timeit import default_timer as timer
import cv2, os

def resize(img):
    # resize size
    return cv2.resize(img, (224, 224))

imgs = []
for file in os.listdir("C://Users//hcaro//OneDrive//Documents//images"):
    img = cv2.imread(os.path.join("C://Users//hcaro//OneDrive//Documents//images",file))
    img = resize(img)
    imgs.append(img)

num_cores = 4;
config = tf.ConfigProto(intra_op_parallelism_threads=8,
                        inter_op_parallelism_threads=8, allow_soft_placement=True,
                        device_count = {'CPU' : 4, 'GPU' : 0})
session = tf.Session(config=config)
ktf.set_session(session)
model = ResNet50(weights='imagenet')

predictions = []
# start time
start = start = timer()
for instance in imgs:
    x = image.img_to_array(instance)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    pred = model.predict(x)
    predictions.append(pred)

for pred in predictions:
    print('Predicted:', decode_predictions(pred, top=3)[0])
# end time
end = timer()
elapsed_time_batch = end-start
print('Runtime: ' + "{0:.2f}".format(end-start) + 's')