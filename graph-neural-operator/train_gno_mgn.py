import torch
import numpy as np
import os
import argparse

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


############################### Training/testing loops ##############################
def train(train_loader, 
          test_loader_list,
          u_normalizer, 
          device, 
          args, 
          kernel_nn_class, 
          PATH=None):

    # Initialize a new wandb run
    wandb.init(project="gno_project", entity="ju1", name=args.model + '_kernel_width' + \
                                               str(args.ker_width) + '_width_'+ str(args.width)+ \
                                                '_depth' + str(args.depth) )
    # Optionally, define the configuration parameters
    wandb.config.update(args)
    
    if args.use_kernel:
        model = kernel_nn_class(args.width,
                     args.ker_width,
                     args.depth,
                     args.edge_features,
                     args.node_features).to(device)
    else:
        # MGN-like model
        model = kernel_nn_class(args.node_features,
                     args.edge_features,
                     args.width,
                     1, args).to(device)

    # Optionally, log model summary
    gnn_model_summary(model, args, args.model + '_depth' + str(args.depth))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    
    myloss = LpLoss(size_average=False)
    u_normalizer.cuda()

    best_test_loss = np.inf
    model.train()

    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        train_mse = 0.0
        train_l2 = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
            # mse.backward()
            loss = torch.norm(out.view(-1) - batch.y.view(-1),1)
            loss.backward()

            l2 = myloss(u_normalizer.decode(out.view(args.batch_size,-1)), u_normalizer.decode(batch.y.view(args.batch_size, -1)))
            # l2.backward()
            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        train_loss = train_l2 / (args.ntrain * args.k)

        # Log training metrics to wandb
        #wandb.log({"epoch": epoch, "train_loss": train_loss, "train_mse": train_mse})

        # Test the model periodically and log the metrics
        if epoch % 10 == 0:
            test_loss_array = test(test_loader_list, args, device, model, u_normalizer, myloss)
            test_loss = np.sum(test_loss_array)
            # Log the max test loss to wandb
            #wandb.log({"epoch": epoch, "combined_test_loss": test_loss})

            # Log each individual test loss
            wandb.log({
                "epoch": epoch,
                "test_loss_1": test_loss_array[0],
                "test_loss_2": test_loss_array[1],
                "test_loss_3": test_loss_array[2]
            })

            # Save the model checkpoint with wandb
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)
                wandb.run.summary["best_test_loss"] = best_test_loss

                # Save the model using the wandb save function
                if PATH is not None:
                    wandb.save(PATH)
        
        # Log the training and test loss to wandb
        wandb.log({"train_loss": round(train_loss, 2), "test_loss": round(test_loss.item(), 2)})
        
        if(epoch%100==0):
            #print("train loss", str(round(total_loss,2)), "test loss", str(round(test_loss.item(),2)))
            if(args.save_best_model):
                PATH = os.path.join(args.checkpoint_dir, args.model + '.pt')
                torch.save(best_model.state_dict(), PATH )
                

    wandb.finish()  # Finish the wandb run
    
    return best_model, best_test_loss


def test(test_loader_list, args, device, model, u_normalizer, myloss):
    '''
    Performs a test loop on the dataset for GNO.
    Explain input arguments:
    test_loader: test dataset
    device: GPU or CPU
    test_model: trained model
    myloss: loss function
    u_normalizer: normalizer for output
    '''
    
    u_normalizer.cuda()
    model = model.to(device)
    ntest = args.ntest
    test_res_size = len(test_loader_list)
    test_loss_array = np.zeros([test_res_size])
    batch_size2 = args.batch_size2
    with torch.no_grad():
        for i in range(3):
            test_loss = 0.0
            for batch in test_loader_list[i]:
                batch.to(device)
                out = model(batch)
                test_l2 = myloss(u_normalizer.decode(out.view(batch_size2,-1)),
                                     batch.y.view(batch_size2, -1))
                test_loss += test_l2.item()
            test_loss_array[i] = test_loss / ntest

    return test_loss_array


############################### Setup hyperparameters ##############################
def main(args):
    # generate a gaussian normalizer
    TRAIN_NAME = 'piececonst_r241_N1024_smooth1.mat'
    reader = MatReader(os.path.join(args.dataset_dir, TRAIN_NAME))
    train_u = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)
    u_normalizer = GaussianNormalizer(train_u)

    # read the preprocessed dataset
    dataset_dir = args.dataset_dir
    file_path=os.path.join(dataset_dir, 'data_train.pt')
    train_loader = DataLoader(torch.load(file_path), batch_size=batch_size, shuffle=True)
    test_loader16 = DataLoader(torch.load(os.path.join(dataset_dir, 'data_test16.pt')), batch_size=batch_size2, shuffle=False)
    test_loader31 = DataLoader(torch.load(os.path.join(dataset_dir, 'data_test31.pt')), batch_size=batch_size2, shuffle=False)
    test_loader61 = DataLoader(torch.load(os.path.join(dataset_dir, 'data_test61.pt')), batch_size=batch_size2, shuffle=False)

    device = args.device if torch.cuda.is_available() else 'cpu'
    #args.device = device
    print('Getting {}...'.format(device))
    args.device = "cpu" # This is necessary for running get_stats function below
    stats_list = stats.get_stats(dataset, args, comp_args)
    print('stats_list: \n{}'.format(stats_list))
    args.device = device

    # Start the training loop
    test_loader_list = [test_loader16, test_loader31, test_loader61]
    best_model, best_test_loss = train(train_loader,
                                    test_loader_list,
                                    u_normalizer, 
                                    device,
                                    args, 
                                    MeshGraphKernel, grid = grid,
                                    PATH=None)


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser()
    
    # Model and data-related arguments
    parser.add_argument('--model', type=str, default='mgk_gno', help='Model identifier.')
    parser.add_argument('--data_type', type=str, default='darcy', help='Type of data.')
    parser.add_argument('--width', type=int, default=64, help='Width parameter.')  # Replace 128 with actual default value
    parser.add_argument('--ker_width', type=int, default=256, help='Kernel width.')  # Replace 256 with actual default value

    # Training process control arguments
    parser.add_argument('--use_kernel', action='store_true', default=True, help='Flag to use kernel.')
    parser.add_argument('--depth', type=int, default=3, help='Depth parameter.')  # Replace 3 with actual default value
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')  # Replace 32 with actual default value
    parser.add_argument('--ntrain', type=int, default=10000, help='Number of training samples.')  # Replace 10000 with actual default value
    parser.add_argument('--k', type=int, default=5, help='K parameter.')  # Replace 5 with actual default value
    parser.add_argument('--ntest', type=int, default=2000, help='Number of test samples.')  # Replace 2000 with actual default value
    parser.add_argument('--batch_size2', type=int, default=2, help='Secondary batch size.')  # Replace 32 with actual default value

    # Feature-related arguments
    parser.add_argument('--edge_features', type=int, default=6, help='Number of edge features.')  # Replace 6 with actual default value
    parser.add_argument('--node_features', type=int, default=6, help='Number of node features.')  # Replace 6 with actual default value

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay rate.')
    parser.add_argument('--scheduler_step', type=int, default=50, help='Scheduler step size.')
    parser.add_argument('--scheduler_gamma', type=float, default=0.8, help='Scheduler gamma value.')

    # Model saving and tracking arguments
    parser.add_argument('--save_best_model', action='store_true', default=True, help='Whether to save the best model.')
    parser.add_argument('--wandb_usr', type=str, default='ju1', help='Weights & Biases user.')
    parser.add_argument('--no_wandb', action='store_true', default=True, help='Flag to not use Weights & Biases.')
    parser.add_argument('--online', action='store_true', default=True, help='Flag for online mode.')

    # Experiment and output directory arguments
    parser.add_argument('--dataset_dir', type=str, default='./datasets/', help='Directory for saving datasets.')
    parser.add_argument('--checkpoint_dir', type=str, default='./best_models/', help='Directory for saving checkpoints.')
    parser.add_argument('--modelsummary_dir', type=str, default='./model_details/', help='Directory for model summaries.')
    parser.add_argument('--postprocess_dir', type=str, default='./2d_loss_plots/', help='Directory for post-processing outputs.')

    
    # random seed
    parser.add_argument('--seed', type=int, help='', default=5)
    args = parser.parse_args()
    
    # setup random seed
    torch.manual_seed(args.seed)  #Torch
    random.seed(args.seed)        #Python
    np.random.seed(args.seed)     #NumPy

    main(args)