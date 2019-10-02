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
from keras.models import load_model
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


# global variables
batch_size       = 16
img_size         = (256, 256, 3)
tr               = pd.read_csv("input/train.csv")
img_names_all    = tr['Image_Label'].apply(lambda x: x.split('_')[0]).unique()
steps_per_epoch  = 200             
epochs           = 20  


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
                    mask = np.zeros((img_size[0], img_size[1]))
                else:
                    mask = rle2mask(rle, img.shape)
                    mask = cv2.resize(mask, (img_size[0], img_size[1]))
                masks.append(mask)                                        
            img = cv2.resize(img, (img_size[0], img_size[1]))            
            x_batch += [img]
            y_batch += [masks] 

            img_names = img_names[img_names != fn]   
        
        x_batch = np.array(x_batch)
        y_batch = np.transpose(np.array(y_batch), (0, 2, 3, 1))        

        yield x_batch, y_batch

def create_model():
    BACKBONE = 'resnet50'
    preprocess_input = sm.backbones.get_preprocessing(BACKBONE)

    model = sm.Unet(
            encoder_name  = BACKBONE, 
            classes       = 4,
            activation    = 'sigmoid',
            input_shape   = img_size)
    
    model.compile(optimizer=optimizers.Adam(lr=9e-3), loss=bce_dice_loss)

    return model


class EpochBegin(keras.callbacks.Callback):
    def on_epoch_begin (self, epoch, logs={}):
        global new_ep
        new_ep = True


def main(model_name):
    ######### TRAINING  #########

    print("\nnumber of segmentation definitions:", len(tr), "(preview seen below)")
    print(tr.head())

    print("\nnumber of unique images:", len(img_names_all))

    print("creating model...")
    model = create_model()

    call_back_list = [EpochBegin()]

    print("training model...")
    model.fit_generator(keras_generator(batch_size),
                        steps_per_epoch  = steps_per_epoch,              
                        epochs           = epochs,                    
                        verbose          = 1,
                        callbacks        = call_back_list)

    print("saving trained model...")
    model.save("models/" + model_name + ".h5")

    gc.collect()

    ######### PREDICTING #########

    # print("loading model")
    # model = load_model("models/" + model_name + ".h5")

    test_images = []
    test_images_paths = glob("input/test/*.jpg") 

    print("reading in test images...")
    for image_path in tqdm(test_images_paths):    
        img = cv2.imread(image_path)
        img = cv2.resize(img,(img_size[0],img_size[1]))       
        test_images.append(img)

    print("length of test images:", len(test_images))

    print("predicting test image segmentations...")
    predict = model.predict(np.asarray(test_images))

    pred_rle = []
    for img in tqdm(predict):      
        img = cv2.resize(img, (525, 350))
        tmp = np.copy(img)
        tmp[tmp<np.mean(img)] = 0
        tmp[tmp>0] = 1
        for i in range(tmp.shape[-1]):
            pred_rle.append(mask2rle(tmp[:,:,i]))

    # print(len(pred_rle))

    # show results of prediction (not currently working)
    # fig, axs = plt.subplots(5, figsize=(20, 20))
    # axs[0].imshow(cv2.resize(plt.imread(test_images_paths[0]),(525, 350)))
    # for i in range(4):
    #     axs[i+1].imshow(rle2mask(pred_rle[i], img.shape))

    # create submission file
    sub = pd.read_csv('sample_submission.csv', converters={'EncodedPixels': lambda e: ' '} )
    sub['EncodedPixels'] = pred_rle
    sub.head()

    sub.to_csv('submissions/submission.csv', index=False)


if __name__ == "__main__":
    main(model_name = "model_1")