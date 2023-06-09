import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import datetime
import os
import random as rn
import shutil
import zipfile
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import random

import tensorflow as tf
from tensorflow.keras.preprocessing import image
import params
import wandb


def aug(df):
    """Takes dataframe as input and Returns dataframe with augmented data as its records"""
    path1 = './Project_data/aug'
    if not os.path.isdir(path1):
            os.mkdir(path1)
    
    train_df = df.reset_index(drop = True).copy()
    train_df = train_df.sample(frac=1,random_state=params.seed)

    aug_dict = {'folder_name':[],'gesture':[], 'label':[], 'folder_path':[]}
    for _,row in tqdm(train_df.iterrows(),total = len(train_df)):
        fold_name, gest, lab = row['folder_name'],row['gesture'],int(row['label'])
        
        frame_list = os.listdir(params.train_image_path+'/'+ (row['folder_name']))
        
        if row['label'] == 0:
            fold_name = 'aug_' + fold_name.replace('Left','Right')
            lab = 1
            gest = 'Right_Swipe_new'
        elif row['label'] == 1:
            fold_name = 'aug_' + fold_name.replace('Right','Left')
            lab = 0
            gest = 'Left Swipe_new'
        else:
             fold_name = 'aug_' + fold_name
        
        path2 = path1 + '/' + fold_name

        if not os.path.isdir(path2) :
            os.mkdir(path2)

        for frame in frame_list:
            image = cv2.imread(params.train_image_path+'/'+ row['folder_name']+'/'+frame).astype(np.float32)
            GaussianBlur = cv2.GaussianBlur(image,(3,3),0)   # Adding gaussian blur
            fliped= cv2.flip(GaussianBlur,1)
            dir_path = path2 + "/"  + frame
            status = cv2.imwrite(dir_path, fliped)
        
        aug_dict['folder_name'].append(fold_name)
        aug_dict['gesture'].append(gest)
        aug_dict['label'].append(lab)
        aug_dict['folder_path'].append(params.train_image_path +'/'+fold_name)

    aug_df = pd.DataFrame(aug_dict)
    aug_df['split'] = 'Train'
    aug_df['IsAugmented'] = True


    source_dir = path1
    target_dir = params.train_image_path

    file_names = os.listdir(source_dir)
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)

    if os.path.exists(path1):
        shutil.rmtree(path1)

    del aug_dict,train_df
    return aug_df


def extract_ds(augment: bool = False):
    """Returns a dataset as a dataframe."""
    if os.path.exists(params.unzipped_file_path):
        shutil.rmtree(params.unzipped_file_path)
    
    print('extracting zipfile:',)
    with zipfile.ZipFile(params.zip_file_path, 'r') as zip_ref:
        zip_ref.extractall('./')
    print('Done extracting:',)

    if os.path.exists('./Project_data/val'):
        os.rename('./Project_data/val', './Project_data/test')
    if os.path.exists('./Project_data/val.csv'):
        os.rename('./Project_data/val.csv', './Project_data/test.csv')
    
    col_names=['folder_name','gesture','label']   

    train_df = pd.read_csv(params.train_csv_path, names = col_names , sep=';')
    train_df['split'] = 'Train'
    train_df['folder_path'] = params.train_image_path + '//' + train_df['folder_name']
    train_df['IsAugmented'] = False
    
    test_df = pd.read_csv(params.test_csv_path, names = col_names , sep=';')
    test_df['split'] = 'Test'
    test_df['folder_path'] = params.test_image_path + '//' + test_df['folder_name']
    test_df['IsAugmented'] = False

    if augment == True:
        print('Augmenting train data:',)
        aug_df = aug(train_df)
        train_df = pd.concat([train_df,aug_df])
        del aug_df
    
    df = pd.concat([train_df,test_df])
    df['gesture'] = df['label'].map(params.BDD_CLASSES)

    num_train_sequences = len(train_df)
    print('# training sequences =', num_train_sequences)
    num_test_sequences = len(test_df)
    print('# test sequences =', num_test_sequences)
    
    del col_names,train_df,test_df
    
    return df



def create_table(df):
    """Create a table from dataframe"""
    table = wandb.Table(columns=["folder_name", "images",'IsAugmented', "gesture", "split","label","folder_path"],allow_mixed_types = True)
    df = df.reset_index(drop = True).copy()
    for i, row in tqdm(df.iterrows(), total=len(df)):
        folder_path = row['folder_path']
        file_names = os.listdir(folder_path)
        img_list = []
        for file in file_names:
            img = image.load_img(folder_path +'\\'+ file)
            img = image.img_to_array(img)
            img = wandb.Image(img)
            img_list.append(img)
        table.add_data(
            str(row['folder_name']),
            img_list,
            row['IsAugmented'],
            row['gesture'],
            row['split'], # we don't have a dataset split yet
            row['label'],
            folder_path
        )

    del img_list
    return table


def generator(df, batch_size, img_size, sampling_type, Shuffle=True, artifact_path = None):
    if sampling_type == 'middle':
        img_idx = list(range(5,25))
    elif sampling_type == 'custom1':
        img_idx = [1,2,3,4,6,8,10,12,14,16,18,20,22,24,26,27,28,29]
    elif sampling_type == 'custom2':
        img_idx = [0,1,2,3,4,5,15,16,17,18,24,25,26,27,28,29]
    elif sampling_type == 'alternate':
        img_idx = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,29]
        
    img_shape = (img_size,img_size,3)

    if Shuffle == True:
        t = np.random.permutation(df)
    else:
        t = df.values

    while True:
        num_batches = len(df)//batch_size
        for batch in range(num_batches): # we iterate over the number of batches
            batch_data = np.zeros((batch_size,len(img_idx),img_shape[0],img_shape[1],img_shape[2])) 
            batch_labels = np.zeros((batch_size,5)) # batch_labels is the one hot representation of the output
            for folder in range(batch_size): # iterate over the batch_size
                path_name = Path(t[folder + (batch*batch_size)][5])
                fold_path = os.path.abspath(os.path.join(artifact_path,path_name.parent.stem,path_name.name))
                frame_list = os.listdir(fold_path) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate over the frames/images of a folder to read them in
                    image = cv2.imread(os.path.join(fold_path,frame_list[item])).astype(np.float32)
                    
                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #since the conv3D will throw error if the inputs in a batch have different shapes
                    
                    image = cv2.resize(image,img_shape[:2],interpolation = cv2.INTER_AREA)   # Resizing the image
                    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) #Normalising the image pixel values
                    
                    
                    batch_data[folder,idx,:,:,0] = image[:,:,0]       #feed in the image
                    batch_data[folder,idx,:,:,1] = image[:,:,1]       #feed in the image 
                    batch_data[folder,idx,:,:,2] = image[:,:,2]       #feed in the image
                    
                    
                batch_labels[folder, int(t[folder + (batch*batch_size)][3])] = 1
                
            yield batch_data, batch_labels #yield the batch_data and the batch_labels

        
        # Code for the remaining data points which are left after full batches
        remaining_datapoints = len(df) % batch_size
        if remaining_datapoints != 0:
            batch += 1
            batch_data = np.zeros((remaining_datapoints,len(img_idx),img_shape[0],img_shape[1],img_shape[2])) 
            batch_labels = np.zeros((remaining_datapoints,5)) # batch_labels is the one hot representation of the output
            for folder in range(remaining_datapoints): # iterate over the batch_size
                path_name = Path(t[folder + (batch*remaining_datapoints)][5])
                fold_path = os.path.abspath(os.path.join(artifact_path,path_name.parent.stem,path_name.name))
                frame_list = os.listdir(fold_path) # read all the images in the folder
                for idx,item in enumerate(img_idx): #  Iterate iver the frames/images of a folder to read them in
                    image = cv2.imread(os.path.join(fold_path,frame_list[item])).astype(np.float32)

                    #crop the images and resize them. Note that the images are of 2 different shape 
                    #and the conv3D will throw error if the inputs in a batch have different shapes
                    image = cv2.resize(image,img_shape[:2],interpolation = cv2.INTER_AREA)  # Resizing the image
                    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) #Normalising the image pixel values

                    batch_data[folder,idx,:,:,0] = image[:,:,0]           #feed in the image
                    batch_data[folder,idx,:,:,1] = image[:,:,1]           #feed in the image
                    batch_data[folder,idx,:,:,2] = image[:,:,2]           #feed in the image
                                        
                batch_labels[folder, int(t[folder + (batch*remaining_datapoints)][3])] = 1
                    
            yield batch_data, batch_labels #yield the batch_data and the batch_labels



def create_table_TargetVsPred(config, model, df, artifact_path, shuffl = False):
    """Compute performance of the model given dataset and log a wandb.Table"""
    btch_size = config.batch_size
    im_size = config.img_size
    samplng_type = config.sampling

    # print(f'config = {config}')
    # print(f'model = {model}')

    if (len(df)%config.batch_size) == 0:
        steps = int(len(df)/config.batch_size)
    else:
        steps = (len(df)//config.batch_size) + 1
    
    summary = {'predicted':[]}

    result_df = df[['folder_name','gesture','label']].copy()
    data_gen = generator(df, batch_size = btch_size, img_size = im_size, sampling_type = samplng_type, Shuffle = shuffl, artifact_path = artifact_path)
    results = np.argmax(model.predict(data_gen,batch_size=config.batch_size,steps=steps), axis=1)
    summary['predicted'].append(results)
    result_df['predicted'] = summary['predicted'].pop()
    accuracy = np.round((len(result_df[result_df.predicted == result_df.label]) / len(result_df)),2)
    table = wandb.Table(dataframe=result_df)
    del summary,result_df
    return table, accuracy

def download_artifact():
    "Grab dataset from artifact"
    processed_data_at = wandb.use_artifact(f'{params.PROCESSED_DATA_AT}:latest')
    metadata = processed_data_at.metadata
    processed_dataset_dir = Path(processed_data_at.download())
    return processed_dataset_dir,metadata


def get_df(processed_dataset_dir_path):
    col =['folder_name', 'split','gesture','label','IsAugmented','folder_path']
    df = pd.read_csv(processed_dataset_dir_path/'data_split.csv', usecols=col)[col]
    df_train = df[df.split == 'Train'].reset_index(drop=True)
    df_valid = df[df.split == 'Valid'].reset_index(drop=True)
    df_test = df[df.split == 'Test'].reset_index(drop=True)
    return df_train,df_valid,df_test

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

