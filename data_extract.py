import os,argparse
from types import SimpleNamespace
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from pathlib import Path
import gc
import numpy as np
import random as rn
# import shutil
# import zipfile
# import csv
from tqdm import tqdm

# from tensorflow.keras.preprocessing import image
import params
import wandb
import helper

ds_config = SimpleNamespace(
    augment=False, # use data augmentation
)


def parse_args():
    "Overriding default argments"
    argparser = argparse.ArgumentParser(description='Dataset extraction parameter')
    argparser.add_argument('--augment', type=bool, default=ds_config.augment, help='Whether to augment or not')
    args = argparser.parse_args()
    vars(ds_config).update(vars(args))
    return


def data_split(df):
    df_train = df[df['split'] == 'Train'].copy()
    df_test = df[df['split'] == 'Test'].copy()
    df_train['fold'] = -1
    cv = StratifiedKFold(n_splits=5)
    for i, (train_idxs, val_idxs) in enumerate(cv.split(df_train['folder_name'].values,
                                                        df_train['label'].values)):
        df_train.loc[val_idxs, ['fold']] = i
    df_train.loc[df_train.fold == 0, ['split']] = 'Valid'
    df_train.drop(columns = 'fold', inplace = True)
    df = pd.concat([df_train,df_test])
    del df_train,df_test
    return df


def wandb_upload(df,config):

    with wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="upload", config=config) as run1:
        print('crating an artifact:')
        raw_data_at = wandb.Artifact(params.RAW_DATA_AT, type="raw_data", metadata = vars(config))

        print('adding original train data to artifact:')
        raw_data_at.add_dir(params.train_image_path, name= 'train')

        print('adding original test data to artifact:')
        raw_data_at.add_dir(params.test_image_path , name= 'test')

        print('creating data table:')
        data_table = helper.create_table(df)

        print('adding data table to artifact:')
        raw_data_at.add(data_table, "Data_table")

        print('logging artifact:')
        run1.log_artifact(raw_data_at)

    with wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="data_split", config=config) as run2:
        print('getting raw data artifact:')
        raw_data_at = run2.use_artifact(f'{params.RAW_DATA_AT}:latest')
        path = Path(raw_data_at.download())
        
        print('splitting the data:')
        df = pd.DataFrame(data=data_table.data, columns=data_table.columns)
        del data_table
        df = data_split(df)
        df.to_csv('data_split.csv', index=False)

        print('crating an artifact:')
        processed_data_at = wandb.Artifact(params.PROCESSED_DATA_AT, type="split_data",metadata = vars(config))

        print('adding data split csv file to artifact:')
        processed_data_at.add_file('data_split.csv')

        print('creating data split table:')
        data_split_table = wandb.Table(dataframe=df[['folder_name', 'split','gesture','label','IsAugmented','folder_path']])
        
        print('adding data split table to artifact:')
        processed_data_at.add(data_split_table, "Data_table_split")

        print('adding original data to artifact:')
        processed_data_at.add_dir(path)

        print('logging artifact:')
        run2.log_artifact(processed_data_at)

        del df,data_split_table

    gc.collect()
    return


if __name__ == '__main__':
    helper.set_seeds(params.seed)
    parse_args()
    print(f'augmenting: {ds_config.augment}')
    df = helper.extract_ds(ds_config.augment)
    wandb_upload(df,ds_config)