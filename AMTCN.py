from numpy import split
from numpy import array
from pandas import read_csv

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import concatenate
from sklearn.preprocessing import MinMaxScaler
import tensorflow.keras
import pickle
import numpy as np
import Multihead attention
import Additive attention

def to_supervised(train, n_input, n_out=24):##The output is 24 or whatever you want.
    # flatten train data
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over the entire history one time step at a time
    for _ in range(len(data)):
        # define the end of the input sequence
        in_end = in_start + n_input  ##0+1*24 day
        out_end = in_end + n_out    ##1+1 day2
        # ensure we have enough data for this instance
        if out_end < len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
        # move along one time step
        in_start += 1
    return array(X), array(y)##X(datalenth-48,24,features num)   y(datalenth-48,24)only 1 features

def BLOCK1(seq, filters):
    cnn = Conv1D(filters*2, 3, padding='same', dilation_rate=1, activation='relu')(seq) ##padding='causal'
    cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
    cnn = Conv1D(filters*2, 3, padding='same', dilation_rate=2, activation='relu')(cnn)
    cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
    cnn = Conv1D(filters*2, 3, padding='same', dilation_rate=4, activation='relu')(cnn)
    cnn = Lambda(lambda x: x[:,:,:filters] + x[:,:,filters:])(cnn)
    if int(seq.shape[-1]) != filters:
        seq = Conv1D(filters, 1, padding='same')(seq)
    seq = add([seq, cnn])
    # seq = LayerNormalization(epsilon=1e-6)(seq)
    return seq

def BLOCK2(seq, filters):
    ##same as 1 but add 1 layer
    # seq = LayerNormalization(epsilon=1e-6)(seq)
    return seq

# train your model
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input) #input output

    # define parameters
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # create a channel for each variable
    in_layers, out_layers = list(), list()
    for i in range(n_features): ##multi head
        inputs = Input(shape=(n_timesteps, 1))
        #inputs=Tensor(None,24,1)

        seq = inputs ##(24,1) #seq=Tensorshape(None,24,1)
        seq = BLOCK1(seq, 8)#16
        seq = BLOCK2(seq, 16)#32
        seq = BLOCK2(seq, 32)#64
        seq = addAttention(units=64)(seq)
        # seq = BatchNormalization()(seq)
        flat = Flatten()(seq)
        in_layers.append(inputs) ##inputs=(None,24,1)
        out_layers.append(flat)

    # merge heads
    merged = concatenate(out_layers)
    merged1 = tensorflow.reshape(merged,shape=[-1,36,16]) ##The shape should be multiplicative to the head_num
    merged1 = MultiHeadAttention(head_num= 8)(merged1) ##2

    merged = Flatten()(merged1)
    dense1 = Dense(200, activation='relu')(merged)##Dense Layer
    dense2 = Dense(100,activation='relu')(dense1)
    outputs = Dense(n_outputs)(dense2)
    model = Model(inputs=in_layers, outputs=outputs)
    # compile model use Adam or else
    adam = tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0, amsgrad=True)
    model.compile(loss='mse', optimizer=adam)#mse

    # fit network
    input_data = [train_x[:, :, i].reshape((train_x.shape[0], n_timesteps, 1)) for i in range(n_features)]
    ##reshape each feature
    ##（train_x.shape[0],n_timesteps,1）），
    return model, input_data

##use CAllBACK