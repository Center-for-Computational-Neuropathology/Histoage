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

    make_map(args.checkpoint, args.h5path, args.wsipath,
            datapath = args.datapath, #path to torch geometric data file, if blank will create one
            outputdir = args.outputdir, #oath to output directory to dump output
            outputbasename = args.outputbasename, #basename to use for output fileneames, if blank will assume h5file basename is sufficient
            argpath = args.argpath, #path to json file with model training options, if blank will assume args.json in parent parent directory of checkpoint file is correct file
            downsample_factor = args.downsample, #how much to downsample the resolution of the WSI
            tile_size = args.tile_size, #The number of pixels in height/width of each tile
            sigma = args.sigma,  # The standard deviation of the Gaussian kernel for smoothing the attention map (makes it look a bit more aesthetically appealing)
            alpha = args.alpha # overlay alpha
            )

    return 
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
## NECESSARY ARGUMENTS
    parser.add_argument('-ckp','--checkpoint', required=True,
                        type=str, help='Path to model checkpoint to extract attention')
    parser.add_argument('-h5','--h5path', required=True,
                        type=str, help='path to h5 file that has coords array and feature vectors')
    parser.add_argument('-s','--wsipath', required=True,
                        type=str, help='path to whole slide image flie (must be openslide compatible)')

## OPTIONAL ARGUMENTS, PATHS
    parser.add_argument('-d','--datapath', default=None,
                        type=str, help='path to torch geometric data file, if blank will create one')
    parser.add_argument('-o','--outputdir', default=None,
                        type=str, help='path to output directory to dump output, if not will create one in same parent directory as checkpoint path')
    parser.add_argument('-b','--outputbasename', default=None,
                        type=str, help='basename to use for output fileneames, if blank will assume h5file basename is sufficient')
    parser.add_argument('-a','--argpath', default=None,
                        type=str, help='path to json file with model training options, if blank will assume args.json in parent parent directory of checkpoint file is correct file')

## OPTIONAL ARGUMENTS, PARAMS
    parser.add_argument('-ds', '--downsample', default=50, type=int,
                        help='how much to downsample the resolution of the WSI')
    parser.add_argument('-ts', '--tile_size', default=256, type=int,
                        help='the number of pixels in height/width of each tile')
    parser.add_argument('-sig','--sigma', default=1.25, type=float,
                        help='the standard deviation of the Gaussian kernel for smoothing the attention map (makes it look a bit more aesthetically appealing)')
    parser.add_argument('-alph','--alpha', default=0.5, type=float,
                        help='the overlay alpha of the attention map ontop of the base slide image, (0,1)')                        

    
    args = parser.parse_args()

    main(args)