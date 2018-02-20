#!/usr/bin/env python
from __future__ import division
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from keras.layers import Input, Dense, Lambda, Layer, Dropout
from keras.models import Model
from keras import backend as K
from keras import metrics
import glob
import pandas
import sys
import argparse
import pickle
import loader

# Parsing the command line #############################
parser = argparse.ArgumentParser(description='Training a VAE with one dense layer',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s','--samples', type=int,   help='Samples per temperature', default=2000)
parser.add_argument('-e','--epochs', type=int,    help='Number of epochs to train', default=50)
parser.add_argument('-d','--dir',                 help='directory with training data' ,default="../samples")
parser.add_argument('--klweight', type=float,     help='Kullback-Leibler weight', default=1)
parser.add_argument('--temps', type=int,          help='number of different temperatures', default=56)
parser.add_argument('--temprep', type=int,        help='number of file per temperature' ,default=2)
parser.add_argument('--size', type=int,           help='number of pixels' ,default=1600)
parser.add_argument('--labels', type=int,         help='number of labels' ,default=2)
parser.add_argument('--batch',  type=int,         help='batch size' ,default=100)
parser.add_argument('--latent', type=int,         help='dimension of latent space' ,default=2)
parser.add_argument('--intermediate', type=int,   help='number of neurons in the dense layer' ,default=256)
parser.add_argument('--epsilon', type=float,      help='standard epsilon' ,default=1.0)
parser.add_argument('--drop', type=float,         help='dropout rate on dense layer' ,default=0)
parser.add_argument('--reconstruction', type=int, help='how many samples to project',default=10000)
parser.add_argument('--read', action='store_true',help='Read arguments from args.pkl')

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

args = parser.parse_args()
if args.read:
    args=load_obj("args")
save_obj(args,'args')

# Setting a few paramters
batch_size = args.batch
original_dim = args.size
latent_dim = args.latent
intermediate_dim = args.intermediate
epochs = args.epochs
epsilon_std = args.epsilon
KLWeight=args.klweight

## Create architecutre  #############################

# Encoder
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
hdrop = Dropout(args.drop)(h)
z_mean = Dense(latent_dim)(hdrop)
z_log_var = Dense(latent_dim)(hdrop)

# Decoder
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon
z = Lambda(sampling)([z_mean, z_log_var])

decoder_h = Dense(intermediate_dim, activation='relu')
decoder_h_drop = Dropout(args.drop)
decoder_mean = Dense(original_dim, activation='sigmoid')

h_decoded = decoder_h(z)
h_decoded_drop = decoder_h_drop(h_decoded)
x_decoded_mean = decoder_mean(h_decoded_drop)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)
        
    def vae_loss(self, x, x_decoded_mean):
        xent_loss =     original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss   = - KLWeight * 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)/(original_dim+KLWeight)
        
    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x
y = CustomVariationalLayer()([x, x_decoded_mean])

## Create model #############################
vae = Model(x, y)
vae.compile(optimizer='adadelta', loss=None)
vae.summary()

exit()

## Train  #############################

# Reading the training files to memory
samples,data,labels=loader.readDirectory(
                args.dir,
                args.temps,
                args.temprep,
                args.samples,
                args.labels,
                args.size)

# prepare data 
permutation=np.random.permutation(samples)

trainingfrac=5/6;
tn=int(trainingfrac*samples);
training = data[permutation][:tn]
validation= data[permutation][tn:]
traininglabels = labels[permutation][:tn]
validationlabels= labels[permutation][tn:]
training = (training-training.min()).astype('float32') / (training.max()-training.min()) 
validation = (validation-validation.min()).astype('float32') / (validation.max()-validation.min())

print training.shape,validation.shape,labels.shape

# do the training
history=vae.fit(training,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(validation,None))

## Creat evaluation models #############################

# build a model to project inputs on the latent space
encoderMu = Model(x, z_mean)
encoderSigma = Model(x, z_log_var)

# build a sample generator that can sample from the learned distribution
decoder_input = Input(shape=(latent_dim,))
_h_decoded = decoder_h(decoder_input)
_x_decoded_mean = decoder_mean(_h_decoded)
generator = Model(decoder_input, _x_decoded_mean)

## Save all we did #############################
generator.save("generator.h5")
encoderSigma.save("encoderSigma.h5")
encoderMu.save("encoderMu.h5")
vae.save("vae.h5")