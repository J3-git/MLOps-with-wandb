import warnings
warnings.filterwarnings("ignore",category=ImportWarning)

import numpy as np
import argparse, os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from types import SimpleNamespace
import gc

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GRU, Flatten, TimeDistributed, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.layers import Conv3D, MaxPooling3D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam,SGD

import params
import helper
import wandb
import data_extract
from wandb.keras import WandbCallback


train_config = SimpleNamespace(
    framework="Keras",
    img_size=64,
    batch_size=8,
    epochs=15, 
    lr=1e-3,
    LR = False,
    optimizer = 'adam',
    sampling = 'custom1',  # whether to use pretrained encoder
    seed=params.seed,
    activation = 'elu',
    dropout = 0.3,
    model = 'conv3d',
    metrics = 'categorical_accuracy',
    augment = data_extract.ds_config.augment,
    total_frames = 18,
    log_preds = False
    )


def parse_args():
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--img_size', type=int, default=train_config.img_size, help='img_size')
    argparser.add_argument('--batch_size', type=int, default=train_config.batch_size, help='batch size')
    argparser.add_argument('--epochs', type=int, default=train_config.epochs, help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=train_config.lr, help='learning rate')
    argparser.add_argument('--seed', type=int, default=train_config.seed, help='random seed')
    argparser.add_argument('--sampling', type=str, default=train_config.sampling, help='middle/custom1/alternate/custom2')
    argparser.add_argument('--optimizer', type=str, default=train_config.optimizer, help='adam/sgd')
    argparser.add_argument('--activation', type=str, default=train_config.activation, help='elu/relu')
    argparser.add_argument('--model', type=str, default=train_config.model, help='conv3d/CnnGRU/resnet50_GRU/VGG16_GRU')
    argparser.add_argument('--dropout', type=float, default=train_config.dropout, help='dropout rate')
    argparser.add_argument('--LR', type=bool, default=train_config.LR, help='ReduceLROnPlateau')
    argparser.add_argument('--log_preds', type=bool, default=train_config.log_preds, help='log predictions?')
    args = argparser.parse_args()
    vars(train_config).update(vars(args))
    return


def Conv3D_model_builder(config):
    
    model = Sequential()
    model.add(Conv3D(64, (3,3,3), strides=(1,1,1), padding='same',
                     input_shape=(config.total_frames,config.img_size,config.img_size,3)))
    model.add(BatchNormalization())
    model.add(Activation(config.activation))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))

    #     model.add(Dropout(config.dropout))

    model.add(Conv3D(128, (3,3,3), strides=(1,1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(config.activation))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))

    model.add(Dropout(config.dropout))

    model.add(Conv3D(256, (3,3,3), strides=(1,1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(config.activation))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))

    model.add(Dropout(config.dropout))

    model.add(Conv3D(256, (3,3,3), strides=(1,1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation(config.activation))
    model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))

    model.add(Flatten())
    model.add(Dropout(config.dropout))
    model.add(Dense(512, activation=config.activation))
    model.add(Dropout(config.dropout))
    model.add(Dense(5, activation='softmax'))
                     
    return model


def CnnGRU_model_builder(config):
    model = Sequential()

    model.add(TimeDistributed(Conv2D(filters=16,kernel_size=(2,2),padding='same', activation=config.activation),
                              input_shape = (config.total_frames,config.img_size,config.img_size,3)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
              
    model.add(TimeDistributed(Dropout(config.dropout)))
              
    model.add(TimeDistributed(Conv2D(filters=32,kernel_size=(2,2),padding='same',activation=config.activation)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
                                     
    model.add(TimeDistributed(Dropout(config.dropout)))

    model.add(TimeDistributed(Conv2D(filters=64,kernel_size=(2,2),padding='same',activation=config.activation)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
                                     
    model.add(TimeDistributed(Dropout(config.dropout)))

    model.add(TimeDistributed(Conv2D(filters=128,kernel_size=(2,2),padding='same',activation=config.activation)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
                                     
    model.add(TimeDistributed(Dropout(config.dropout)))

    model.add(TimeDistributed(Conv2D(filters=256,kernel_size=(2,2),padding='same',activation=config.activation)))
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    model.add(TimeDistributed(Dropout(config.dropout)))

    # model.add(TimeDistributed(Conv2D(filters=512,kernel_size=(2,2),padding='same',activation=config.activation)))
    # model.add(TimeDistributed(BatchNormalization()))
    # model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

    # model.add(TimeDistributed(Dropout(config.dropout)))
                                     
    model.add(TimeDistributed(Flatten()))
    model.add(GRU(512))
    model.add(Dropout(config.dropout))
    model.add(Dense(256,activation=config.activation))
    model.add(Dropout(config.dropout))
    model.add(Dense(128,activation=config.activation))
    model.add(Dropout(config.dropout))
    model.add(Dense(5,activation='softmax'))
                    
    return model


def resnet50_GRU_model_builder(config):
    resnet = ResNet50(include_top=False,weights='imagenet',
                      input_shape=(config.img_size,config.img_size,3))
    cnn =Sequential([resnet])
    cnn.add(Conv2D(16,(2,2),strides=(1,1),padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(config.dropout))
    cnn.add(Conv2D(16,(3,3),strides=(1,1),padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(Dropout(config.dropout))
    cnn.add(Flatten())

    model_Resnet50GRU = Sequential()
    model_Resnet50GRU.add(TimeDistributed(cnn,
                                          input_shape = (config.total_frames,config.img_size,config.img_size,3)))
    model_Resnet50GRU.add(GRU(32,input_shape=(None,config.total_frames,256),return_sequences=True))
    model_Resnet50GRU.add(GRU(16))
    model_Resnet50GRU.add(Dropout(config.dropout))
    model_Resnet50GRU.add(Dense(5,activation='softmax'))
    
    return model_Resnet50GRU


def VGG16_GRU_model_builder(config):
    base_model = VGG16(include_top=False,weights='imagenet',input_shape=(config.img_size,config.img_size,3))

    for layer in base_model.layers:
        layer.trainable = False

    base_model_ouput = base_model.output
    x = Flatten()(base_model_ouput)
    features_1 = Dense(128, activation=config.activation)(x)
    # features_2 = Dropout(0.4)(features_1)
    features_3 = Dense(64, activation=config.activation)(features_1)
    features_4 = Dropout(config.dropout)(features_3)
    init_model = Model(inputs=base_model.input, outputs=features_4)

    model_Vgg16GRU = Sequential()
    model_Vgg16GRU.add(TimeDistributed(init_model,
                                       input_shape=(config.total_frames,config.img_size,config.img_size,3)))
    model_Vgg16GRU.add(GRU(32,return_sequences=True))
    model_Vgg16GRU.add(GRU(16))
    model_Vgg16GRU.add(Dropout(config.dropout))
    model_Vgg16GRU.add(Dense(8, activation=config.activation))
    model_Vgg16GRU.add(Dropout(config.dropout))
    model_Vgg16GRU.add(Dense(5,activation='softmax'))
    
    return model_Vgg16GRU


def train(config):
    # tf.random.set_seed(config.seed)
    helper.set_seeds(config.seed)
    if config.sampling == 'custom1':
        total_frames = 18
    elif config.sampling == 'middle':
        total_frames = 20
    elif config.sampling == 'custom2':
        total_frames = 16
    elif config.sampling == 'alternate':
        total_frames = 16

    vars(config).update({'total_frames':total_frames})

    with wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="training", config=config) as run:
        processed_dataset_dir,meta = helper.download_artifact()
        df_train,df_valid,_ = helper.get_df(processed_dataset_dir)
        # good practice to inject params using sweeps
        wandb.config.update(meta, allow_val_change=True)
        config = wandb.config

        num_train_sequences = len(df_train)
        num_val_sequences = len(df_valid)
        
        # if config.sampling == 'custom1':
        #     total_frames = 18
        # elif config.sampling == 'middle':
        #     total_frames = 20
        # elif config.sampling == 'custom2':
        #     total_frames = 16
        # elif config.sampling == 'alternate':
        #     total_frames = 16

        # vars(train_config).update({'total_frames':total_frames})
        # vars(config).update({'total_frames':total_frames})

        train_gen = helper.generator(df_train, config.batch_size, config.img_size, config.sampling, artifact_path = processed_dataset_dir)
        val_gen = helper.generator(df_valid, config.batch_size, config.img_size, config.sampling, artifact_path = processed_dataset_dir)
        
        if config.model == 'conv3d':
            model = Conv3D_model_builder(config)
        elif config.model == 'CnnGRU':
            model = CnnGRU_model_builder(config)
        elif config.model == 'resnet50_GRU':
            model = resnet50_GRU_model_builder(config)
        elif config.model == 'VGG16_GRU':
            model = VGG16_GRU_model_builder(config)

        if config.optimizer == 'sgd':
            optimiser = SGD(learning_rate=config.lr)
        elif config.optimizer == 'adam':
            optimiser = Adam(learning_rate=config.lr)

        if config.LR == True:
            LR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, mode='min', epsilon=0.0001, cooldown=0, min_lr=0.00001)
            callbacks = [WandbCallback(log_model=True),LR]
        else:
            callbacks = [WandbCallback(log_model=True)]
        
        model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=[config.metrics])
        
        if (num_train_sequences%config.batch_size) == 0:
            steps_per_epoch = int(num_train_sequences/config.batch_size)
        else:
            steps_per_epoch = (num_train_sequences//config.batch_size) + 1


        if (num_val_sequences%config.batch_size) == 0:
            validation_steps = int(num_val_sequences/config.batch_size)
        else:
            validation_steps = (num_val_sequences//config.batch_size) + 1


        hist = model.fit(train_gen, steps_per_epoch=steps_per_epoch, epochs=config.epochs, verbose=1,
                  callbacks=callbacks, validation_data=val_gen, validation_steps=validation_steps,
                  class_weight=None, workers=1, initial_epoch=0)
        
        wandb.summary['val_accuracy'] = np.round(hist.history['val_categorical_accuracy'][-1],2)
        wandb.summary['train_accuracy'] = np.round(hist.history['categorical_accuracy'][-1],2)
        
        if config.log_preds == True:
            TargetVsPred,_ = helper.create_table_TargetVsPred(config, model, df_valid, artifact_path = processed_dataset_dir, shuffl = False)
            run.log({'TargetVsPred' : TargetVsPred})

        del model,df_train,df_valid
        tf.keras.backend.clear_session()
        gc.collect()

    
    return

if __name__ == '__main__':
    warnings.filterwarnings("ignore",category=ImportWarning)
    # helper.set_seeds(train_config.seed)
    parse_args()
    train(train_config)