import warnings
warnings.filterwarnings("ignore")

import segmentation_models as sm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import random
from tqdm import tqdm_notebook
import cv2
import gc

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


tr = pd.read_csv("input/train.csv")

print(len(tr))
tr.head()