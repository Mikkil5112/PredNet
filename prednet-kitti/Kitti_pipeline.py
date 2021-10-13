# -*- coding: utf-8 -*-
"""
Created on Sat May  1 11:49:36 2021

@author: mikki
"""

import os
import sys
import time
import glob
import argparse
import json

from multiprocessing import Pool
from functools import partial
from six.moves import cPickle

import numpy as np
import pandas as pd
from PIL import Image as im

import skimage
#import cupy as cp
from skimage.feature import structure_tensor, structure_tensor_eigvals
#from structure_tensor import eig_special_2d, structure_tensor_2d

import keras
from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten, Lambda, TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

#HDF5_DISABLE_VERSION_CHECK=2
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR)
print(tf.test.is_gpu_available())
print(tf.config.list_physical_devices('GPU'))

import matplotlib
# import StringIO
# import urllib, base64
# import statmodels.api as sm

matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg

np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)