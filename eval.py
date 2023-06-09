import wandb
import params
import helper
from pathlib import Path
import os,argparse
import tensorflow as tf
from tensorflow import keras
from types import SimpleNamespace
import warnings

split_config = SimpleNamespace(
    split='eval', # use eval set
)

def parse_args():
    argparser = argparse.ArgumentParser(description='evalution and inference')
    argparser.add_argument('--split', type=str, default=split_config.split, help='eval/test')
    args = argparser.parse_args()
    vars(split_config).update(vars(args))
    return

def mapping(config):
    job_type = 'evalution' if config.split == 'eval' else 'inference'
    return str(job_type)


def evalution(config,job):
    with wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type=job, tags = ['staging']) as run:
        artifact = run.use_artifact('j3-wandb/model-registry/gesture_final:v0', type='model')
        artifact_dir = Path(artifact.download())
        print(f'artifact_dir: {artifact_dir}')
        print(f'artifact_dir content: {os.listdir(artifact_dir)}')

        producer_run = artifact.logged_by()
        wandb.config.update(producer_run.config)
        wandb.config.update({'log_preds':True}, allow_val_change=True)
        print(f'producer_run config: {producer_run.config}')
        print(f'wandb config: {wandb.config}')

        _model_pth = os.listdir(artifact_dir)[1]
        print(f'model path: {_model_pth}')

        # model_path = _model_pth.parent.absolute()/_model_pth.stem
        # model_path = model_path.parent
        # model_path = os.path.join(model_path,"")
        # print(f'complete model path: {model_path}')

        processed_dataset_dir,_ = helper.download_artifact()
        _,df_valid, df_test = helper.get_df(processed_dataset_dir)

        model = keras.models.load_model(artifact_dir)

        if config.split == 'eval':
            TargetVsPred_valid,val_acc = helper.create_table_TargetVsPred(wandb.config, model, df_valid, artifact_path = processed_dataset_dir, shuffl = False)
            wandb.summary['valid_acc'] = val_acc
            run.log({'TargetVsPred_valid' : TargetVsPred_valid})
        else:
            TargetVsPred_test,test_acc = helper.create_table_TargetVsPred(wandb.config, model, df_test, artifact_path = processed_dataset_dir, shuffl = False)
            wandb.summary['test_acc'] = test_acc
            run.log({'TargetVsPred_test' : TargetVsPred_test})
    
    return


if __name__ == '__main__':
    warnings.filterwarnings("ignore",category=ImportWarning)
    # helper.set_seeds(train_config.seed)
    parse_args()
    job = mapping(split_config)
    evalution(split_config,job)


