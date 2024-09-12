import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys, argparse, os, copy, itertools, glob, datetime
import pandas as pd
import numpy as np


import torch
import torch_geometric
import tempfile
import h5py
import libpysal

import time
import math

class GraphDataset(torch_geometric.data.Dataset):
    """ Dataset for .h5 files created from
        CLAM with coordinates and features, given a csv of cases to include.
        Will loop through each cases to a corresponding
        list of paths. Those paths are then used for getitem.
    """


    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform=None, pre_transform=None)
        # self.k = k #number of tiles to establish neighbour radius
        # self.df = pd.read_csv(self.raw_paths[0])
        # self.ids = [os.path.basename(i).split('.')[0] for i in self.df['h5file'].values] 
        
    @property
    def raw_file_names(self):
        return 'h5files_age.csv'
    
    @property
    def processed_file_names(self):
        df = pd.read_csv(self.raw_paths[0])
        ids = [os.path.basename(i).split('.')[0] for i in df['h5file'].values]

        return  [ f'{i}.pt' for i in ids ] #[f'data_{i}.pt' for i in range(len(pd.read_csv(self.raw_paths[0])))]
    

    def len(self):
        return len(self.processed_file_names)


    def process(self):
        tilesize = 256
        k = 1 #number of tiles to establish neighbour radius
        self.r = math.ceil(k * tilesize * math.sqrt(2))
        self.df = pd.read_csv(self.raw_paths[0])
        for i,row in self.df.iterrows():

            age = row['AGE']
            h5file = row['h5file']
            y = torch.tensor([age]).view(1,1)
            f = h5py.File(h5file,'r')
            X = torch.tensor(f['features'])

            coords = f['coords']
            w = libpysal.weights.DistanceBand(coords, self.r, binary=True, silence_warnings=True)
            edges = w.to_adjlist()
            edgelist = torch.LongTensor(edges[['focal', 'neighbor']].values.T)

            data = torch_geometric.data.Data(x=X, y=y, edge_index= edgelist)
            data.num_nodes = X.shape[0]
            id = os.path.basename(h5file).split('.')[0]
            torch.save(data, os.path.join(self.processed_dir, f'{id}.pt'))
    def get(self, idx):
        df = pd.read_csv(self.raw_paths[0])
        h5file = df.iloc[idx,1]
        id = os.path.basename(h5file).split('.')[0]
        data = torch.load(os.path.join(self.processed_dir, '{}.pt'.format(id)))
        return data
