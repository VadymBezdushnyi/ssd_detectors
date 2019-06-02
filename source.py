# In[0]


# In[1]
import cv2
import keras
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import Callback
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time
import glob
import pickle

from sl_model import SL512, DSODSL512
from ssd_data import InputGenerator
from ssd_data import preprocess
from sl_utils import PriorUtil
from ssd_utils import load_weights, calc_memory_usage
from ssd_training import Logger, LearningRateDecay
from sl_training import SegLinkLoss, SegLinkFocalLoss

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (8, 8)
plt.rcParams['image.interpolation'] = 'nearest'

np.set_printoptions(suppress=True, linewidth=120)

# In[2]
from data_synthtext import GTUtility

file_name = 'gt_util_synthtext_seglink.pkl'
with open(file_name, 'rb') as f:
    gt_util = pickle.load(f)
gt_util_train, gt_util_val = gt_util.split(gt_util, split=0.9)
gt_util_train, _ = gt_util.split(gt_util_train, split=0.25)
gt_util_val, _ = gt_util.split(gt_util_val, split=0.25)

print(gt_util)

# In[3]
from data_synthtext import GTUtility

file_name = 'gt_util_synthtext_seglink.pkl'
with open(file_name, 'rb') as f:
    gt_util = pickle.load(f)
gt_util_train, gt_util_val = gt_util.split(gt_util, split=0.9)
gt_util_train, _ = gt_util.split(gt_util_train, split=0.25)
gt_util_val, _ = gt_util.split(gt_util_val, split=0.25)

print(gt_util_train)

# In[4]
# SegLink + DenseNet
model = DSODSL512()
#model = DSODSL512(activation='leaky_relu')
weights_path = None
freeze = []
batch_size = 6
experiment = 'dsodsl512_synthtext'

# In[5]
prior_util = PriorUtil(model, model.source_layers_names)
image_size = model.image_size

if weights_path is not None:
    load_weights(model, weights_path)

# In[6]
inputs = []
images = []
data = []

gtu = gt_util_val

np.random.seed(1337)

for i in np.random.randint(0, gtu.num_samples, 16):

    img_path = os.path.join(gtu.image_path, gtu.image_names[i])
    img = cv2.imread(img_path)
    
    inputs.append(preprocess(img, image_size))
    
    h, w = image_size
    img = cv2.resize(img, (w,h), cv2.INTER_LINEAR).astype('float32') # should we do resizing
    img = img[:, :, (2,1,0)] # BGR to RGB
    img /= 255
    images.append(img)
    
    boxes = gtu.data[i]
    data.append(boxes)

inputs = np.asarray(inputs)

test_idx = 0
test_input = inputs[test_idx]
test_img = images[test_idx]
test_gt = data[test_idx]

#plt.figure()
#plt.imshow(test_img)
#gt_util.plot_gt(test_gt, show_labels=False)
#plt.show()

# In[7]
epochs = 100
initial_epoch = 0

gen = InputGenerator(gt_util_train, gt_util_val, prior_util, batch_size, image_size)

for layer in model.layers:
    layer.trainable = not layer.name in freeze

checkdir = './checkpoints/' + time.strftime('%Y%m%d%H%M') + '_' + experiment
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

with open(checkdir+'/source.py','wb') as f:
    source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
    f.write(source.encode())

optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0)
#optim = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0, nesterov=True)

# weight decay
regularizer = keras.regularizers.l2(5e-4) # None if disabled
#regularizer = None
for l in model.layers:
    if l.__class__.__name__.startswith('Conv'):
        l.kernel_regularizer = regularizer

#loss = SegLinkLoss(lambda_offsets=1.0, lambda_links=1.0, neg_pos_ratio=3.0)
loss = SegLinkFocalLoss()

model.compile(optimizer=optim, loss=loss.compute, metrics=loss.metrics)

history = model.fit_generator(
        gen.generate(train=True, augmentation=False), #generator, 
        #gen.generate(train=True, augmentation=True), #generator, 
        gen.num_train_batches, 
        epochs=epochs, 
        verbose=1, 
        callbacks=[
            keras.callbacks.ModelCheckpoint(checkdir+'/weights.{epoch:03d}.h5', verbose=1, save_weights_only=True),
            Logger(checkdir),
            #LearningRateDecay()
        ], 
        validation_data=gen.generate(train=False, augmentation=False), 
        validation_steps=gen.num_val_batches, 
        class_weight=None,
        max_queue_size=1, 
        workers=1, 
        #use_multiprocessing=False, 
        initial_epoch=initial_epoch, 
        #pickle_safe=False, # will use threading instead of multiprocessing, which is lighter on memory use but slower
        )

