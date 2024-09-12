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

    
    torch.manual_seed(0)
    device = torch.device('cuda')
    os.makedirs(args.dump_path,exist_ok=True)
    jsonpath = os.path.join(args.dump_path,'args.json')
    with open(jsonpath, 'w') as f:
        json.dump(vars(args), f)
    total_summary_csv = os.path.join(args.dump_path,'overall_summary.csv')
    dataset = GraphDataset(args.data)
    slide_ids = [os.path.basename(i).split('.')[0] for i in dataset.processed_paths]
    datalist = [torch.load(i) for i in dataset.processed_paths]
    ages = [int(i.y.numpy()) for i in datalist]
    ranks = np.ones_like(ages)
    if args.stratify:
        ranks[ages > np.quantile(ages,0.25)] = 2
        ranks[ages > np.quantile(ages,0.50)] = 3
        ranks[ages > np.quantile(ages,0.75)] = 4
    splits = args.n_splits

    for k in range(args.start,splits):
        print(f'********************************** STARTING SPLIT {k+1} **********************************', flush=True)
        split_out = os.path.join(args.dump_path,str(k))
        os.makedirs(split_out,exist_ok=True)

        summary_csv = os.path.join(split_out,'training_summary.csv')

        test_dataset,train_dataset = train_test_split(datalist, test_size=args.split, stratify = ranks)

        train_loader = DataLoader(train_dataset,
                                batch_size= args.batch,
                                shuffle= True, # want to shuffle the dataset
                                num_workers=1) # number processes/CPUs to use
        

        test_loader = DataLoader(test_dataset,
                                batch_size= args.batch,
                                shuffle= False,
                                num_workers=1)         

        all_loader = DataLoader(datalist,
                                batch_size= args.batch,
                                shuffle= False,
                                num_workers=1) 

        LR = args.learning
        REG = args.reg
        feat_dim = train_dataset[0].x.shape[1]

        # Model creation
        if args.network == 'trans':
            model = graph_set_transformer(L=feat_dim,
                                  hidden_dim = args.hdim, 
                                  dropout= args.dropout, 
                                  Conv = SAGEConv, 
                                  layer_norm=args.layernorm, 
                                  nheads=args.nheads,
                                  mab_fc=args.mab_fc)


        elif args.network == 'set_mean':
                
            model = graph_set_mean(L=feat_dim,
                                  hidden_dim = args.hdim, 
                                  dropout= args.dropout, 
                                  Conv= SAGEConv)
        else:
            raise ValueError('available networks are trans or set_mean')


        model = model.to(device)


        if args.optim == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=REG)
        elif args.optim =='sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=REG, momentum = args.momentum)
        elif args.optim =='adamw':
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=REG)

        criterion = nn.MSELoss()


        print('Model created, beginning training.')

        training_data = {'train_loss' : [],
                        'train_rmse': [],
                        'val_loss': [],
                        'val_rmse': []}
        tracker = []

        counter = 0
        best_score = None
        for epoch in range(0, args.n_epochs):
            row = dict()
            row['epoch'] = epoch
            print(f"=========================   SPLIT: {k+1} EPOCH: {epoch+1} COUNTER: {counter}  =========================" , flush=True)
            model, train_loss, elapsed, train_rmse, train_mae = train(train_loader,
                                                    model,
                                                    criterion,
                                                    optimizer,
                                                    device,
                                                    args)
            training_data['train_loss'].append(train_loss)
            training_data['train_rmse'].append(train_rmse)
            row['train_loss'] = train_loss
            row['train_rmse'] = train_rmse
            row['train_mae'] = train_mae
            

            print(f"Epoch {epoch+1} Train Set\tElapsed: {elapsed:.1f}\tLoss: {train_loss:.4f}\tRMSE: {train_rmse:.4f}", flush=True)
            val_loss, val_elapsed, val_rmse, val_mae, preds, truths = evaluate(test_loader,
                                                                      model,
                                                                      criterion,
                                                                      device)
            
            training_data['val_loss'].append(val_loss)
            training_data['val_rmse'].append(val_rmse)
            row['val_loss'] = val_loss
            row['val_rmse'] = val_rmse
            row['val_mae'] = val_mae
            score = -val_loss
            if best_score == None:
                best_score = score
                best_rmse = val_rmse
                torch.save(model.state_dict(), os.path.join(split_out, "checkpoint.pth"))
            elif score < best_score:
                counter += 1
            else:
                best_score = score
                print('New best val RMSE -- saving model.')
                counter = 0
                torch.save(model.state_dict(), os.path.join(split_out, "checkpoint.pth"))
                best_row = row
                best_rmse = val_rmse
                joint = sns.jointplot(x=truths,y=preds)
                joint.set_axis_labels('True Age','Predicted Age')
                min_age = np.min(np.concatenate((truths,preds)))
                max_age = np.max(np.concatenate((truths,preds)))
                joint.ax_marg_x.set_xlim(min_age-5, max_age+5)
                joint.ax_marg_y.set_ylim(min_age-5, max_age+5)
                joint.savefig(os.path.join(split_out,'distribution.png'))
                plt.close()


            tracker.append(row)            
            print(f"Val Set\tElapsed: {val_elapsed:.1f}\tLoss: {val_loss:.4f}\tRMSE: {val_rmse:.4f}\tLast best:{best_rmse:.4f}", flush=True)
            if epoch % 10 == 0:
                df = pd.DataFrame(tracker)
                df.to_csv(summary_csv)
                make_performance_figure(split_out,df)


        
        model.load_state_dict(torch.load(os.path.join(split_out, "checkpoint.pth")))
        test_loss, test_elapsed, test_rmse, test_mae, preds, truths = evaluate(test_loader,
                                                                    model,
                                                                    criterion,
                                                                    device)
        
        test_df = pd.DataFrame()
        test_df['truth'] = truths
        test_df['predicted'] = preds
        test_df['residuals'] = truths - preds
        test_df.to_csv(os.path.join(split_out,'test_results.csv'))

        all_df = pd.DataFrame()
        all_df['slide'] = slide_ids
        all_df['AGE'] = ages
        all_loss, all_elapsed, all_rmse, all_mae, preds, truths = evaluate(all_loader,
                                                                    model,
                                                                    criterion,
                                                                    device)
        all_df['truth'] = truths
        all_df['predicted'] = preds
        all_df['residuals'] = truths - preds
        all_df.to_csv(os.path.join(split_out,'all-cases_results.csv'))

        if os.path.exists(total_summary_csv):
            ovr_trkr = pd.read_csv(total_summary_csv)
            ovr_trkr = ovr_trkr.drop(['Unnamed: 0'],axis=1)
            overall_tracker = ovr_trkr.to_dict('records')
        else:
            overall_tracker = []
        split_tracker = {'split': k,
                         'test_rmse': test_rmse,
                         'test_mae': test_mae,
                         'test_loss': test_loss,
                         'val_rmse': best_row['val_rmse'],
                         'val_mae': best_row['val_mae'],
                         'val_loss': best_row['val_loss'],
                         'train_rmse': best_row['train_rmse'],
                         'train_mae': best_row['train_mae'],
                         'train_loss': best_row['train_loss']}
        overall_tracker.append(split_tracker)
        analysis_summary = pd.DataFrame(overall_tracker)


        analysis_summary.to_csv(total_summary_csv)
        
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
        fig.savefig(os.path.join(split_out, 'test_results.png' ))
        plt.close()

    

        print(f"*************************\tSPLIT {k+1} FINISHED TRAINING AFTER {epoch} EPOCHS\t*************************", flush=True)        
        print(f"*************************\tTEST LOSS: {test_loss:.4f} TEST RMSE: {test_rmse:.4f}\t*************************", flush=True)        
        print(f"*************************\tVAL LOSS: {val_loss:.4f} VAL RMSE: {val_rmse:.4f}\t*************************", flush=True)        
        print(f"*************************\tTRAIN LOSS: {train_loss:.4f} TRAIN RMSE: {train_rmse:.4f}\t*************************", flush=True)        

        plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

## write/load directory/file paths
    parser.add_argument('-d','--data', default= '/sc/arion/projects/tauomics/age_regression/graph/data/lhe/IN1024_k1',
                        type=str,help='Path to data directory (based in torch geometric format)')
    parser.add_argument('--dump_path', default='.', type=str,
                        help='path to store experiment information in.')


## Model structure options
    parser.add_argument('-net','--network', default='trans', choices=['trans','set_mean'],
                        type=str, help='which model archetection you want to use.')
    parser.add_argument('-hd', '--hdim', default=256,
                        type=int, metavar='hidden layer dimension size')        
    parser.add_argument('-nh', '--nheads', default=2, type=int,
                        help='number of heads for multiheaded attention')
    parser.add_argument('--mab_fc', action='store_true', default=False)
    parser.add_argument('--layernorm', action='store_true', default=False)


## Regularization hyperparameters
    parser.add_argument('-lr', '--learning', default=1e-4,
                        type=float, metavar='learning rate')    
    parser.add_argument('-r', '--reg', default=1e-5,
                        type=float, metavar='regularization')
    parser.add_argument('-do', '--dropout', default=0.25,
                        type=float, metavar='dropout')
    parser.add_argument('-mo', '--momentum', default=0.9,
                        type=float, metavar='momentum (only works if optimizer is SGD)')
    parser.add_argument('-opt', '--optim', default='adamw', choices=['adam','sgd','adamw'],
                        type=str, help="optimizer to choose: Adam ('adam'), AdamW ('adamw'), or stochastic gradient descent ('sgd')")
    parser.add_argument('-b', '--batch', default=8, type=int,
                        help='pseudo batch size for training, will update model every batch_size iterations.')


## Training parameters
    parser.add_argument('--split', default=.90, type=float,
                        help='Fraction of dataset to use for training set')
    parser.add_argument('--n_splits', default=50, type=int,
                        help='Number of splits to run')
    parser.add_argument('--stratify', default='store_true', default=False,
                        help='Stratify the distribution of ages for train/test split such that the train and test sets will have roughly the same distribution of ages')
    parser.add_argument('--n_epochs', default=100, type=int,
                        help='Number of epochs for each split')    
    parser.add_argument('--start', default=0, type=int,
                        help='K-fold that training has been started at (used if restarting).')
    
    args = parser.parse_args()




    main(args)