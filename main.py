import warnings
warnings.filterwarnings("ignore")

import segmentation_models as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from tqdm import tqdm
import cv2
import gc
from glob import glob

import albumentations as albu
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, UpSampling2D, Conv2D, Activation
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.losses import binary_crossentropy
from keras.callbacks import Callback, ModelCheckpoint
from keras import optimizers

from sklearn.model_selection import train_test_split


def rle2mask(rle, imgshape):
    width = imgshape[0]
    height= imgshape[1]
    
    mask= np.zeros( width*height ).astype(np.uint8)
    
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start+lengths[index])] = 1
        current_position += lengths[index]
        
    return np.flipud( np.rot90( mask.reshape(height, width), k=1 ) )


def mask2rle(img):
    pixels= img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


new_ep = False
def keras_generator(batch_size):  
    global new_ep
    while True:   
        
        x_batch = []
        y_batch = []        
        for _ in range(batch_size):                         
            if new_ep == True:
                img_names =  img_names_all
                new_ep = False
            
            fn = img_names[random.randrange(0, len(img_names))]                                       

            img = cv2.imread('input/train/'+ fn)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                       
            masks = []
            for rle in tr[tr['Image_Label'].apply(lambda x: x.split('_')[0]) == fn]['EncodedPixels']:                
                if pd.isnull(rle):
                    mask = np.zeros((img_size, img_size))
                else:
                    mask = rle2mask(rle, img.shape)
                    mask = cv2.resize(mask, (img_size, img_size))
                masks.append(mask)                                        
            img = cv2.resize(img, (img_size, img_size))            
            x_batch += [img]
            y_batch += [masks] 

            img_names = img_names[img_names != fn]   
        
        x_batch = np.array(x_batch)
        y_batch = np.transpose(np.array(y_batch), (0, 2, 3, 1))        

        yield x_batch, y_batch

class EpochBegin(keras.callbacks.Callback):
    def on_epoch_begin (self, epoch, logs={}):
        global new_ep
        new_ep = True


img_size = 256
tr = pd.read_csv("input/train.csv")

print(len(tr))
print(tr.head())

img_names_all = tr['Image_Label'].apply(lambda x: x.split('_')[0]).unique()
print(len(img_names_all))

BACKBONE = 'resnet50'
preprocess_input = sm.backbones.get_preprocessing(BACKBONE)

model = sm.Unet(
           encoder_name=BACKBONE, 
           classes=4,
           activation='sigmoid',
           input_shape=(img_size, img_size, 3))

model.compile(optimizer=optimizers.Adam(lr=9e-3), loss=bce_dice_loss)

Epoch_Begin_Clb = EpochBegin()

batch_size = 16
# model.fit_generator(keras_generator(batch_size),
#               steps_per_epoch=200,              
#               epochs=20,                    
#               verbose=1,
#               callbacks=[Epoch_Begin_Clb]
#               )

gc.collect()

test_img = []
# testfiles=os.listdir('input/test/*.jpg')
testfiles = glob("input/test/*.jpg") 
print("--------------------here------------------------")
for image_path in tqdm(testfiles):    
    # print(fn)
    # input() 
    img = cv2.imread(image_path)
    img = cv2.resize(img,(img_size,img_size))       
    test_img.append(img)

print(len(test_img))

predict = model.predict(np.asarray(test_img))

pred_rle = []
for img in predict:      
    img = cv2.resize(img, (525, 350))
    tmp = np.copy(img)
    tmp[tmp<np.mean(img)] = 0
    tmp[tmp>0] = 1
    for i in range(tmp.shape[-1]):
        pred_rle.append(mask2rle(tmp[:,:,i]))

print(len(pred_rle))

# show results of prediction
fig, axs = plt.subplots(5, figsize=(20, 20))
axs[0].imshow(cv2.resize(plt.imread(testfiles[0]),(525, 350)))
for i in range(4):
    axs[i+1].imshow(rle2mask(pred_rle[i], img.shape))

# create submission file
sub = pd.read_csv('sample_submission.csv', converters={'EncodedPixels': lambda e: ' '} )
sub['EncodedPixels'] = pred_rle
sub.head()

sub.to_csv('submissions/submission.csv', index=False)