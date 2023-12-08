import torch
import numpy as np
import os
import argparse
import random
import itertools

import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LayerNorm
from tqdm import trange
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
from utilities import *
import nn_conv
from kernel_models import *

import pandas as pd
import copy
import wandb

import visualizer as viz

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

############################### Training/testing loops ##############################
def train(train_loader, 
          test_loader_list,
          stats_list, 
          eval_datasets,
          device, 
          args, comp_args,
          PATH=None):

    # Initialize a new wandb run
    wandb.init(project=args.project_name, entity="ju1", name=args.model + '_' +args.data_type + '_kernel_width' + \
                                               str(args.ker_width) + '_width_'+ str(args.width)+ \
                                                '_depth' + str(args.depth)+'_ntrain' + str(args.ntrain) + \
                                                '_equEdge' + str(args.use_agu_edge))
    # Optionally, define the configuration parameters
    wandb.config.update(args)
    
    # Dictionary mapping model names to class constructors
    model_classes = {
        'gno': KernelNN,
        'dgno': DeepKernelNN,
        'mgo': MeshGraphKernel,
        'nkn': NolocalKernelNN,
        'mgn': MeshGraphNet,
        'ino': KernelNN
    }
    
    # Instantiate the selected model class
    if args.model.lower() in model_classes:
        if args.use_kernel:
            # kernel method
            model = model_classes[args.model.lower()](args.width,
                         args.ker_width,
                         args.depth,
                         args.node_features,
                         args.edge_features, args.output_features).to(device)
        else:
            # MGN-like model
            model = model_classes[args.model.lower()](args.node_features,
                         args.edge_features,
                         args.width,
                         1, args).to(device)

        print(f'Model {args.model} initialized.')
    else:
        raise ValueError(f'Model {args.model} not recognized.')
    
    
    # Optionally, log model summary
    gnn_model_summary(model, args, args.model + '_' + args.data_type +'_depth' + str(args.depth))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    
    myloss = LpLoss(size_average=False)
    #The statistics of the data are decomposed
    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
        std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_y.to(device),std_vec_y.to(device))
    
    best_test_loss = np.inf
    best_train_loss = np.inf
    test_batch_list = []
    train_batch = None
    model.train()
    
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        train_rmse = 0.0
        train_sh_rmse = 0.0
        num_loops = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            if (args.noise):
                # Injecting noise is only used for one-step model
                # perturb (input, output) pairs with a zero-mean gaussian distribution
                # current verison adopts a hard-coded noise_scale (0.003), used in deepmind
                zero_size = torch.zeros(batch.x[:, 0].size(), dtype=torch.float32)
                noise = torch.normal(zero_size, std=args.noise_scale)
                # saturation
                batch.x[:, 0] += noise
                batch.y[:, 0] += noise

            out = model(batch, mean_vec_x,std_vec_x,mean_vec_edge,
                                            std_vec_edge) # needs to rescale data
            #mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
            # mse.backward()
            #loss = torch.norm(out.view(-1) - batch.y.view(-1),1)
            loss = model.loss(out, batch, mean_vec_y, std_vec_y)
            loss.backward()
            
            # here needs to scale
            model_sh = unnormalize( out.squeeze(), mean_vec_y[0], std_vec_y[0] )
            gs_sh = batch.y[:, 0].squeeze()
            error_sh = (model_sh - gs_sh)** 2
            sh_rmse = torch.sqrt( torch.mean(error_sh) )

            #l2 = myloss(unnormalize(out.squeeze(), mean_vec_y, std_vec_y),
            #            unnormalize(batch.y.squeeze(), mean_vec_y, std_vec_y))
        
            # l2.backward()
            optimizer.step()
            train_rmse += loss.item()
            train_sh_rmse += sh_rmse.item()
            num_loops += 1
            
        scheduler.step()
        train_rmse /= num_loops
        train_sh_rmse /= num_loops

        # Log training metrics to wandb
        #wandb.log({"epoch": epoch, "train_loss": train_loss, "train_mse": train_mse})

        # Test the model periodically and log the metrics
        if epoch % args.test_freq == 0:
            test_loss_array, test_eval_loss_array = test(test_loader_list, args, device, model, stats_list, myloss)
            test_loss = np.sum(test_loss_array)
            test_eval_loss = np.sum(test_loss_array)
            # Log the max test loss to wandb
            #wandb.log({"epoch": epoch, "combined_test_loss": test_loss})

            # Log each individual test loss
            wandb.log({
                "test_loss_1": test_loss_array[0],
                "test_loss_2": test_loss_array[1],
                "test_eval_loss_1": test_eval_loss_array[0],
                "test_eval_loss_2": test_eval_loss_array[1]                 # superresolution
            })
            
            # Save the model checkpoint with wandb
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)
                wandb.run.summary["best_test_loss"] = best_test_loss

                # Save the model using the wandb save function
                if PATH is not None:
                    wandb.save(PATH)
                    
        if epoch % args.eval_freq == 0:
            
            if train_rmse < best_train_loss:
                # This might be useful to see how overfitting the model is
                best_train_loss = train_rmse
                best_train_model = copy.deepcopy(model)
                wandb.run.summary["best_train_loss"] = best_train_loss

                # Save the model using the wandb save function
                if PATH is not None:
                    wandb.save(PATH)
            
            # visualize model with best test loss
            test_anim_caption_list = ['test35_model_epoch{}'.format(epoch), 'test50_model_epoch{}'.format(epoch)]
            for i, caption in enumerate(test_anim_caption_list):
                viz.visualize_one_step_rollout(args, best_model, eval_datasets[i], stats_list, comp_args,
                                               caption, 0, wandb, mode=caption[:6])
            
            # visualize model with best test loss
            train_anim_model = 'train35_model_epoch{}'.format(epoch)
            viz.visualize_one_step_rollout(args, best_train_model, eval_datasets[2], stats_list, comp_args,
                                           train_anim_model, args.mesh_num, wandb, mode="train35")
            
        # Log the training and test loss to wandb
        wandb.log({"train_eval_loss": round(train_sh_rmse, 5),
                   "train_loss": round(train_rmse, 5),
                   "test_loss": round(test_loss, 5),
                  "test_eval_loss": round(test_eval_loss, 5)})
        
        if(epoch%100==0):
            #print("train loss", str(round(total_loss,2)), "test loss", str(round(test_loss.item(),2)))
            if(args.save_best_model):
                PATH = os.path.join(args.checkpoint_dir, args.model + '.pt')
                torch.save(best_model.state_dict(), PATH )
                

    wandb.finish()  # Finish the wandb run
    return best_model, best_test_loss

def test(test_loader_list, args, device, model, stats_list, myloss):
    '''
    Performs a test loop on the dataset for GNO.
    Explain input arguments:
    test_loader: test dataset
    device: GPU or CPU
    test_model: trained model
    myloss: loss function
    u_normalizer: normalizer for output
    '''
    [mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y] = stats_list
    (mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y)=(mean_vec_x.to(device),
        std_vec_x.to(device),mean_vec_edge.to(device),std_vec_edge.to(device),mean_vec_y.to(device),std_vec_y.to(device))
    
    model = model.to(device)
    test_res_size = len(test_loader_list)
    test_loss_array = np.zeros([test_res_size]) # test loss 
    test_eval_loss_array = np.zeros([test_res_size]) # evaluation loss; rescaled to physical units
    batch_size2 = args.batch_size2
    with torch.no_grad():
        for i in range(test_res_size):
            test_loss = 0.0
            test_eval_loss = 0.0
            loop_num = 0

            for batch in test_loader_list[i]:
                batch.to(device)
                out = model(batch)
                #test_l2 = myloss(unnormalize(out.squeeze(), mean_vec_y, std_vec_y),
                #        unnormalize(batch.y.squeeze(), mean_vec_y, std_vec_y))
                
                loss = model.loss(out, batch, mean_vec_y, std_vec_y, num = 0) # rmse of scaled data
                # evaluation loss
                model_sh = unnormalize( out.squeeze(), mean_vec_y[0], std_vec_y[0] )
                gs_sh = batch.y[:, 0].squeeze()
                error_sh = (model_sh - gs_sh)** 2
                sh_rmse = torch.sqrt(torch.mean(error_sh))
                
                test_loss += loss.item()
                test_eval_loss += sh_rmse.item()
                loop_num += 1
                
            test_loss_array[i] = test_loss / loop_num
            test_eval_loss_array[i] = test_eval_loss / loop_num

    return test_loss_array, test_eval_loss_array

############################### Setup hyperparameters ##############################
def main(args, comp_args):
    # generate a gaussian normalizer
    
    # read the preprocessed dataset
    dataset_dir = args.dataset_dir
    file_names = ['meshPEBI_train35_datahomo_coord_varsat_modelMGO_totalTs19_skip5_multistep1_distedge_ylabel_nonerelPerm.pt',
                  'meshPEBI_test50_datahomo_coord_varsat_modelMGO_totalTs19_skip5_multistep1_distedge_ylabel_nonerelPerm.pt']
    
    if args.use_agu_edge:
        # use equivarience gnn by augmenting dataset
        file_names = [f"aug_{name}" for name in file_names]

    # Load data using the selected file names
    # be careful these data has not been scaled, needs to be scaled
    train_dataset = torch.load(os.path.join(dataset_dir, file_names[0]))
    train_loader35 = DataLoader(train_dataset[:args.ntrain*args.rollout_num], batch_size=args.batch_size, shuffle=False)
    test_loader35 = DataLoader(train_dataset[args.ntrain*args.rollout_num:(args.ntrain + args.ntest)*args.rollout_num], batch_size=args.batch_size, shuffle=False)
    
    test_dataset = torch.load(os.path.join(dataset_dir, file_names[1]))
    test_loader50 = DataLoader(test_dataset, batch_size=args.batch_size2, shuffle=False)
    train35_stats_list = get_stats(train_dataset, args, comp_args, use_single_dist = False)
    
    eval_train35_dataset = train_dataset[args.mesh_num*args.rollout_num: (args.mesh_num+1)*args.rollout_num]
    eval_test35_dataset = train_dataset[(args.ntrain + args.mesh_num)*args.rollout_num: (args.ntrain + args.mesh_num+1)*args.rollout_num]
    eval_test50_dataset = test_dataset[args.mesh_num*args.rollout_num: (args.mesh_num+1)*args.rollout_num]
    eval_datasets = [ eval_test35_dataset, eval_test50_dataset, eval_train35_dataset]
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    #args.device = device
    print('Getting {}...'.format(device))
    
    # Start the training loop: super resolution
    test_loader_list = [test_loader35, test_loader50]
    best_model, best_test_loss = train(train_loader35,
                                    test_loader_list,
                                    train35_stats_list,  # contain the scaling factors for all node and edge features
                                    eval_datasets,
                                    device,
                                    args, comp_args, 
                                    PATH=None)


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    
    # Model and data-related arguments
    parser.add_argument('--model', type=str, default='mgn', help='Model identifier.')
    parser.add_argument('--data_type', type=str, default='pebi_fractured', help='Type of data.')
    parser.add_argument('--width', type=int, default=64, help='Width parameter.')  # Replace 128 with actual default value
    parser.add_argument('--ker_width', type=int, default=200, help='Kernel width.')  # Replace 256 with actual default value
    parser.add_argument('--use_agu_edge',action='store_true', default=False, help='Flag to use equivarience-augemented edge features.')
    parser.add_argument('--noise', action='store_true', help='', default=False) # inject noise for one-step predictions
    parser.add_argument('--noise_scale', type=float, help='', default=0.003) # noise scale
    parser.add_argument('--rollout_num', type=int, default=19, help='Number of rollout.')
    parser.add_argument('--well_weight', type=float, help='', default=0.700)
    parser.add_argument('--node_type_index', type=int, help='', default=6) # the starting index of node type, used for locating speical nodes
                                                                               # current version: [x, y, sh, perm , poro, volume, type0, type1, type2, type3]
    
    # Training process control arguments
    parser.add_argument('--use_kernel', action='store_true', default=False, help='Flag to use kernel.')
    parser.add_argument('--depth', type=int, default=3, help='Depth parameter.')  # Replace 3 with actual default value
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size.')  # Replace 32 with actual default value
    parser.add_argument('--ntrain', type=int, default=400, help='Number of training meshes.')  # Replace 10000 with actual default value
    parser.add_argument('--k', type=int, default=1, help='Number of output.')  # Replace 5 with actual default value
    parser.add_argument('--r', type=int, default=4, help='Radius of stencil.')  # Replace 5 with actual default value
    
    parser.add_argument('--ntest', type=int, default=50, help='Number of test meshes.')  # Replace 2000 with actual default value
    parser.add_argument('--batch_size2', type=int, default=1, help='Secondary batch size.')  # Replace 32 with actual default value
    
    # Feature-related arguments
    parser.add_argument('--edge_features', type=int, default=3, help='Number of edge features.')  # Replace 6 with actual default value
    parser.add_argument('--node_features', type=int, default=10, help='Number of node features.')  # [x, y, sh, perm , poro, volume, type0, type1, type2, type3]
    parser.add_argument('--output_features', type=int, default=1, help='Number of label features.')  # Replace 6 with actual default value
    parser.add_argument('--rela_perm', type=str, default='none', help='Flag for using relative permeability as edge feature')
    parser.add_argument('--var_type', type=str, default='sat', help='Physical meaning of output label')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay rate.')
    parser.add_argument('--scheduler_step', type=int, default=50, help='Scheduler step size.')
    parser.add_argument('--scheduler_gamma', type=float, default=0.8, help='Scheduler gamma value.')

    # Model saving and tracking arguments
    parser.add_argument('--save_best_model', action='store_true', default=True, help='Whether to save the best model.')
    parser.add_argument('--wandb_usr', type=str, default='ju1', help='Weights & Biases user.')
    parser.add_argument('--no_wandb', action='store_true', default=True, help='Flag to not use Weights & Biases.')
    parser.add_argument('--online', action='store_true', default=True, help='Flag for online mode.')
    parser.add_argument('--project_name', type=str, default='gno_project', help='wandb project names.')
    parser.add_argument('--test_freq', type=int, default=10, help='Interval for evaluating test cases.')
    parser.add_argument('--eval_freq', type=int, default=50, help='Interval for evaluating rollout results.')
    parser.add_argument('--mesh_num', type=int, default=0, help='Mesh number for demonstration.')
    
    # Experiment and output directory arguments
    parser.add_argument('--dataset_dir', type=str, default='./datasets/', help='Directory for saving datasets.')
    parser.add_argument('--checkpoint_dir', type=str, default='./best_models/', help='Directory for saving checkpoints.')
    parser.add_argument('--modelsummary_dir', type=str, default='./model_details/', help='Directory for model summaries.')
    parser.add_argument('--postprocess_dir', type=str, default='./2d_loss_plots/', help='Directory for post-processing outputs.')
    parser.add_argument('--device', type=str, help='', default='cuda') # cuda could vary from cuda:0 to cuda:3 depending on how many avaialble GPUs
    
    # random seed
    parser.add_argument('--seed', type=int, help='', default=5)
    args = parser.parse_args()
    
    # setup random seed
    torch.manual_seed(args.seed)  #Torch
    random.seed(args.seed)        #Python
    np.random.seed(args.seed)     #NumPy
    
    # physics data enhancement
    # computational arguments
    for c_args in [
            {'is_initial':True,
            'coord_sg':'',
             'coord_sw':'',
             'rel_sg':'',
             'rel_sw':'', },
        ]:
            comp_args = objectview(c_args)

    main(args, comp_args)