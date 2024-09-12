import sys, argparse, os, copy, itertools, glob, datetime, math, json, posixpath, time

import warnings

import torch_geometric
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms.functional as VF
from torchvision import transforms

from torch_geometric.nn import DenseSAGEConv, GCNConv, GINConv, SAGEConv, dense_diff_pool,  Sequential,GENConv, DeepGCNLayer, GraphConv,GINConv, TopKPooling, MessagePassing
from torch_geometric.utils import to_dense_adj, to_dense_batch, degree
from torch_geometric.nn.pool.topk_pool import topk,filter_adj
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.data import DataLoader
import libpysal
import h5py

import argparse, os, socket
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

import pandas as pd
import numpy as np
from sklearn import metrics
import seaborn as sns
import openslide
from scipy.ndimage import gaussian_filter
import PIL 

from age_gnn import *


def train(loader, model, criterion, optim, device, args):
    model.train()

    start_t = time.perf_counter()
    epoch_loss = 0
    truths = []
    preds = []
    for it, data in enumerate(loader):
        data = data.to(device)
        Y_hat = model(data)
        age = data.y.to(device)
        loss = criterion(Y_hat, age.type(torch.float))
        pred_age = Y_hat
        epoch_loss += loss.item()
        loss.backward()

        truths.extend(age.cpu().numpy().flatten())
        preds.extend(pred_age.detach().cpu().numpy().flatten())


        optim.step()
        model.zero_grad()

    preds = np.array(preds)
    truths = np.array(truths)
    epoch_loss /= len(loader)
    rmse = metrics.mean_squared_error(truths, preds, squared=False)
    mae  = metrics.mean_absolute_error(truths, preds)
    elapsed_time = time.perf_counter() - start_t
    return model, epoch_loss, elapsed_time, rmse, mae


def evaluate(loader, model, criterion, device):
    model.eval()

    start_t = time.perf_counter()
    epoch_loss = 0
    truths = []
    preds = []
    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            Y_hat = model(data)
            age = data.y.to(device)

            loss = criterion(Y_hat, age.type(torch.float))
            pred_age = Y_hat
            epoch_loss += loss.item()

            truths.extend(age.cpu().numpy().flatten())
            preds.extend(pred_age.detach().cpu().numpy().flatten())

        preds = np.array(preds)
        truths = np.array(truths)
        epoch_loss /= len(loader)
        rmse = metrics.mean_squared_error(truths, preds, squared=False)
        mae  = metrics.mean_absolute_error(truths, preds)
        elapsed_time = time.perf_counter() - start_t
    return epoch_loss, elapsed_time, rmse, mae, preds, truths

def make_performance_figure(split_out, df_):
    df = df_.copy()
    df['log_train_loss'] = np.log(df['train_loss'])
    df['log_val_loss'] = np.log(df['val_loss'])

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex='col')
    fig.tight_layout()
    df.plot(x='epoch',y=['train_loss','val_loss'],ax=axes[0])
    axes[0].set_title('Loss')
    axes[0].legend(loc='upper left')

    df.plot(x='epoch',y=['log_train_loss','log_val_loss'],ax=axes[1])
    axes[1].set_title('Log MSE')
    axes[1].legend().remove()

    df.plot(x='epoch',y=['train_rmse','val_rmse'],ax=axes[2])
    axes[2].set_title('RMSE')
    axes[2].legend().remove()

    fig.savefig(os.path.join(split_out, 'training_performance.png' ))
    plt.close()
    return

def init_model(jsonpath, feat_dim=1024):
    with open(jsonpath,'r') as f:
        args = json.load(f)
    
        # Model creation
    if args['network'] == 'trans':
        model = graph_set_transformer(L=feat_dim,
                                hidden_dim = args['hdim'], 
                                dropout= args['dropout'], 
                                Conv = SAGEConv, 
                                layer_norm=args['layernorm'], 
                                nheads=args['nheads'],
                                mab_fc=args['mab_fc'])


    elif args['network'] == 'set_mean':
            
        model = graph_set_mean(L=feat_dim,
                                hidden_dim = args['hdim'], 
                                dropout= args['dropout'], 
                                Conv= SAGEConv)
    else:
        raise ValueError('available networks are trans or set_mean')

    return model

def h5file_to_geomdata(h5file, tilesize=256, k=1, age=None):
    #k is number of tiles to establish neighbour radius
    r = math.ceil(k * tilesize * math.sqrt(2))
    if age==None:
        y=torch.empty(1,1)
    else:
        y = torch.tensor([age]).view(1,1)
    f = h5py.File(h5file,'r')
    X = torch.tensor(f['features'])
    coords = f['coords']
    w = libpysal.weights.DistanceBand(coords, r, binary=True, silence_warnings=True)
    edges = w.to_adjlist()
    edgelist = torch.LongTensor(edges[['focal', 'neighbor']].values.T)
    data = torch_geometric.data.Data(x=X, y=y, edge_index= edgelist)
    
    return data


def make_map(modelcheckpoint,
            h5path,
            wsipath,
            datapath=None, #path to torch geometric data file, if blank will create one
            outputdir=None, #oath to output directory to dump output
            outputbasename=None, #basename to use for output fileneames, if blank will assume h5file basename is sufficient
            argpath=None, #path to json file with model training options, if blank will assume args.json in parent parent directory of checkpoint file is correct file
            downsample_factor = 50, #how much to downsample the resolution of the WSI
            tile_size = 256, #The number of pixels in height/width of each tile
            sigma = 1.25,  # The standard deviation of the Gaussian kernel for smoothing the attention map (makes it look a bit more aesthetically appealing)
            alpha = 0.5 # overlay alpha
            ):

    #assuming you can run this on cpu, there is no model training so should be all good!
    device = torch.device('cpu') 

    # load json containing the training options
    if argpath==None:
        argpath = posixpath.join(posixpath.abspath(posixpath.join(modelcheckpoint,'../..')),'args.json')
    with open(argpath,'r') as f:
        args = json.load(f)

    # load h5file
    f = h5py.File(h5path,'r')
    coords = f['coords']

    # creating torch geometric data object if necessary 
    if datapath == None:
        dataobj = h5file_to_geomdata(h5path)
    else:
        dataobj = torch.load(datapath)
    
    # establishing outpath names
    if outputbasename==None: # if outputbasename is not defined, assuming can use same basename as datafile
        datafilepath, ext = os.path.splitext(h5path)
        outputbasename = os.path.basename(datafilepath) 
    if outputdir==None: #if outputdir not defined, assume can place in same dir as model checkpoint
        outputdir = posixpath.join(posixpath.abspath(posixpath.join(modelcheckpoint,'../')),'attn_maps')
    if not os.path.isdir(outputdir):
        os.makedirs(outputdir,exist_ok=True)
    slideout = os.path.join(outputdir,outputbasename+'_orig.png')
    
    # load model
    feat_dim = dataobj.x.shape[1]
    model = init_model(argpath,feat_dim)
    model.load_state_dict(torch.load(modelcheckpoint,map_location=device))
    model.eval()

    # extract attention weights for each tile 
    dataloader = DataLoader([dataobj])
    data = next(iter(dataloader))
    data, model = data.to(device), model.to(device)
    with torch.no_grad():
        y, A, A_raw = model(data, return_attn=True)

    #process the WSI
    wsi = openslide.OpenSlide(wsipath) #loads the WSI into a openslide object
    w,h = wsi.dimensions
    w_downsampled, h_downsampled = int(w/downsample_factor), int(h/downsample_factor)
    # Determine the number of tiles in the x and y directions
    num_tiles_x = (w + tile_size - 1) // tile_size
    num_tiles_y = (h + tile_size - 1) // tile_size

    # Create a PIL image from the downsampled WSI
    wsi_downsampled = wsi.get_thumbnail((w_downsampled, h_downsampled))
    wsi_downsampled.save(slideout)
    wsi_array = np.array(wsi_downsampled.convert('RGB'))
    
    for a in range(args['nheads']+1): #looping through each attention head and then after that, making an averaged attention map
        if a == args['nheads']:
            attn = A_raw.mean(0)
            mapfile = os.path.join(outputdir,outputbasename+'_mean_attn.png')
        else:
            attn = A_raw[a,0,:]
            mapfile = os.path.join(outputdir,outputbasename+f'_{str(a)}_attn.png')
        
        labels = attn.squeeze().cpu().numpy()
        labels = (labels - labels.mean())/labels.std() #standardizing the attention values (important for converting to colormap)

        # Create an array to store the attention values at the tile resolution
        attention_values_tile = np.zeros((num_tiles_y, num_tiles_x))

        # Loop over the coordinates and labels to fill in the attention_values array at the tile resolution
        for coord, label in zip(coords, labels):
            x, y = int(coord[0]), int(coord[1])
            tile_x, tile_y = x // tile_size, y // tile_size
            attention_values_tile[tile_y, tile_x] = label
        
        # apply gaussian filter to smooth the attention values        
        attention_values_tile = gaussian_filter(attention_values_tile, sigma=sigma)

        # Convert the attention values to colors using the jet colormap
        colors = cm.jet(attention_values_tile)
        colors = Image.fromarray(np.uint8(colors*255)).resize((w_downsampled, h_downsampled),resample= PIL.Image.NEAREST)
        colors_array = np.array(np.asarray(colors))

        # Create a new numpy array to store the overlayed image
        overlayed = np.zeros((h_downsampled, w_downsampled, 3), dtype=np.uint8)
        # # Create the mask based on the shape of the overlayed array
        mask = np.sum(colors_array[:, :, :2],2)==0

        # Loop over the color array and the overlayed array to create the overlayed image
        for y in range(1, h_downsampled-1):
            for x in range(1, w_downsampled-1):
                if mask[y,x]:
                    overlayed[y,x,:] = wsi_array[y-1, x-1, :]
                else:
                    overlayed[y,x,:] = (colors_array[y, x, :3]  * alpha) + (wsi_array[y-1, x-1, :] * (1 - alpha))

        image = Image.fromarray(overlayed)
        image.save(mapfile)
    
    return