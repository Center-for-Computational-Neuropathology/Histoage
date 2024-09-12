import sys, argparse, os, copy, itertools, glob, datetime, math, json, posixpath, time

from tarfile import CONTTYPE
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch_geometric.data import DataLoader
from torch_geometric.nn import DenseSAGEConv, GCNConv, SAGEConv, DenseGCNConv
import argparse, os, socket
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sns
import time

from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split


from age_gnn import *

def main(args):

    dataset = GraphDataset(args.data)
    modelcheckpoint = args.checkpoint
    argpath = args.argpath
    outputdir = args.outputdir
    device = args.device

    # load json containing the training options
    if argpath==None:
        argpath = posixpath.join(posixpath.abspath(posixpath.join(modelcheckpoint,'../..')),'args.json')
    with open(argpath,'r') as f:
        args = json.load(f)
    slide_ids = [os.path.basename(i).split('.')[0] for i in dataset.processed_paths]
    datalist = [torch.load(i) for i in dataset.processed_paths]
    ages = [int(i.y.numpy()) for i in datalist]

    if outputdir==None: #if outputdir not defined, assume can place in same dir as model checkpoint
        outputdir = posixpath.join(posixpath.abspath(posixpath.join(modelcheckpoint,'../')),'eval')
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir,exist_ok=True)

    dataloader = DataLoader(dataset,
                            batch_size= 1,
                            shuffle= False,
                            num_workers=1)         

    criterion = nn.MSELoss()
    feat_dim = dataset[0].x.shape[1]
    model = init_model(argpath,feat_dim)
    model.load_state_dict(torch.load(modelcheckpoint,map_location=device))
    model.eval()

    test_loss, elapsed_time, test_rmse, test_mae, preds, truths = evaluate(dataloader, model, criterion, device)

    test_df = pd.DataFrame()
    test_df['slide'] = slide_ids
    test_df['AGE'] = ages
    test_df['truth'] = truths
    test_df['predicted'] = preds
    test_df['residuals'] = truths - preds
    test_df.to_csv(os.path.join(outputdir,'results.csv'))
    min_age = np.min(np.concatenate((truths,preds)))
    max_age = np.max(np.concatenate((truths,preds)))
    fig, axes = plt.subplots(nrows=4, ncols=1,figsize=(8,9))
    axes[0].set_xlim(min_age-5,max_age+5)
    sns.scatterplot(data=test_df,x='truth', y='residuals', ax=axes[0])
    sns.regplot(data=test_df,x='truth', y='predicted', ax=axes[1])
    sns.histplot(data=test_df,x='truth',ax=axes[2],kde=True)
    sns.histplot(data=test_df,x='predicted',ax=axes[3],kde=True)
    fig.suptitle(f"Loss: {test_loss:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
    axes[0].set_ylabel('Residuals')
    axes[1].set_ylabel('Predicted Age')
    axes[2].set_ylabel('')
    axes[3].set_ylabel('')
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    axes[2].set_xlabel('True Age')
    axes[3].set_xlabel('Predicted Age')
    axes[1].set_xlim(axes[0].get_xlim())
    axes[2].set_xlim(axes[0].get_xlim())
    axes[3].set_xlim(axes[0].get_xlim())
    axes[1].set_ylim(axes[0].get_xlim())
    fig.tight_layout()
    fig.savefig(os.path.join(outputdir, 'results_fig.png' ))
    plt.close()

    joint = sns.jointplot(x=truths,y=preds)
    joint.set_axis_labels('True Age','Predicted Age')
    joint.ax_marg_x.set_xlim(min_age-5, max_age+5)
    joint.ax_marg_y.set_ylim(min_age-5, max_age+5)
    joint.savefig(os.path.join(outputdir,'distribution.png'))
    plt.close()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

## NECESSARY ARGUMENTS
    parser.add_argument('-d','--data', required=True,
                        type=str,help='Path to data directory (based in torch geometric format)')
    parser.add_argument('-ckp','--checkpoint', required=True,
                        type=str, help='Path to model checkpoint to evaluate')

## OPTIONAL ARGUMENTS, PATHS   
    parser.add_argument('-o','--outputdir', default=None,
                        type=str, help='path to output directory to dump output, if not will create one in same parent directory as checkpoint path')
    parser.add_argument('-a','--argpath', default=None,
                        type=str, help='path to json file with model training options, if blank will assume args.json in parent parent directory of checkpoint file is correct file')
    parser.add_argument('--device', default='cpu',
                        type=str, help='device to run the model on for eval. likely options are "cuda" (to run on gpu) or "cpu" (to run locally)')

    args = parser.parse_args()
    main(args)