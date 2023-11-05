import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LayerNorm
from utilities import *
import nn_conv

import importlib
importlib.reload(nn_conv) # for debugging purposes
from nn_conv import NNConv_old
from nn_conv import ProcessorLayer
from nn_conv import OperatorProcessorLayer

from nn_conv import NNConv_NKN
from egcl import E_GCL_GKN
from timeit import default_timer


#################################################
#
# all graph-based kernel models
#
#################################################

class KernelNN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN, self).__init__()
        self.depth = depth
        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet([ker_in, ker_width, ker_width, width**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        #print('x shape {}'.format(x.shape))
        for k in range(self.depth):
            #print('x shape {} at depth {}'.format(x.shape, k))
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            
        x = self.fc2(x)
        return x


class MeshGraphKernel(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1):
        super(MeshGraphKernel, self).__init__()
        """

        Input_dim: dynamic variables + node_type (node_position is encoded in edge attributes)
        Hidden_dim: 128 in deepmind's paper
        Output_dim: dynamic variables: velocity changes (1)

        """

        self.num_layers = args.depth
        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(Linear(in_width, width),
                              ReLU(),
                              Linear( width, width),
                              LayerNorm(width))
        
        # edge and node has the same width
        self.edge_encoder = Sequential(Linear( in_width , width),
                              ReLU(),
                              Linear( width, width),
                              LayerNorm(width)
                              )

        self.processor = nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'
        #self.processor.append(self.conv1)
        
        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            kernel = DenseNet([width, ker_width//2, ker_width//2, width**2], torch.nn.ReLU)
            self.processor.append(processor_layer(width, width, kernel, aggr='mean'))

        # decoder: only for node embeddings
        self.decoder = Sequential(Linear( width , width),
                              ReLU(),
                              Linear( width, out_width)
                              )
        
    def build_processor_model(self):
        return OperatorProcessorLayer


    def forward(self, data):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension

        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x,edge_attr = self.processor[i](x,edge_index,edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest

        return self.decoder(x)



class MeshGraphNet(torch.nn.Module):
    def __init__(self, 
                 input_dim_node, 
                 input_dim_edge, 
                 hidden_dim, 
                 output_dim, args, emb=False):
        super(MeshGraphNet, self).__init__()
        """

        Input_dim: dynamic variables + node_type (node_position is encoded in edge attributes)
        Hidden_dim: 128 in deepmind's paper
        Output_dim: dynamic variables: velocity changes (1)

        """

        self.num_layers = args.depth

        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(Linear(input_dim_node , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim))

        self.edge_encoder = Sequential(Linear( input_dim_edge , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, hidden_dim),
                              LayerNorm(hidden_dim)
                              )


        self.processor = nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'

        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim,hidden_dim))


        # decoder: only for node embeddings
        self.decoder = Sequential(Linear( hidden_dim , hidden_dim),
                              ReLU(),
                              Linear( hidden_dim, output_dim)
                              )


    def build_processor_model(self):
        return ProcessorLayer


    def forward(self, data):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x) # output shape is the specified hidden dimension

        edge_attr = self.edge_encoder(edge_attr) # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x,edge_attr = self.processor[i](x,edge_index,edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest

        return self.decoder(x)



class NolocalKernelNN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1,batch_size = 2,grid = None):
        super(NolocalKernelNN, self).__init__()
        self.depth = depth
        #print("in_width: %d" % in_width)
        self.fc1 = torch.nn.Linear(in_width, width)
        self.Bx = DenseNet([width, ker_width//2, ker_width, width**2], torch.nn.ReLU) 
        kernel = DenseNet([ker_in, ker_width//2, ker_width, width**2], torch.nn.ReLU)
        self.conv1 = NNConv_NKN(width, width, kernel, aggr='mean')
        #self.grid = grid.repeat(batch_size,1).to(device)
        #print(self.grid)
        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        
        for k in range(self.depth):
            # No reaction terms
            #x = F.relu(self.conv1(x, edge_index, edge_attr)/self.depth)
            # reaction terms
            #print('x shape {}'.format(x.shape))
            x = self.conv1(x, edge_index, edge_attr)/self.depth + \
                 (torch.matmul(self.Bx(x).view(-1,width,width), x.unsqueeze(2)).squeeze()-x)/self.depth + x
        #x = self.conv2(x, edge_index, edge_attr)
        #x = self.conv3(x, edge_index, edge_attr)
        #x = self.conv4(x, edge_index, edge_attr)
        x = self.fc2(x)
        return x


class EGKN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1, act_fn=torch.nn.ReLU()):
        super().__init__()
        self.depth = depth
        
        self.fc1 = torch.nn.Linear(in_width, width)
        kernel = DenseNet([ker_in, ker_width // 2, ker_width, width ** 2], torch.nn.ReLU)
        self.egkn_conv = E_GCL_GKN(width, width, width, kernel, depth, act_fn=act_fn)
        self.fc2 = torch.nn.Sequential(torch.nn.Linear(width, width * 2), act_fn, torch.nn.Linear(width * 2, out_width))

    def forward(self, data):
        h, edge_index, edge_attr, coords_curr = data.x, data.edge_index, data.edge_attr, \
                                                data.coords_init.detach().clone()
        h = self.fc1(h)
        for k in range(self.depth):
            h, coords_curr = self.egkn_conv(h, edge_index, coords_curr, edge_attr)
        
        h = self.fc2(h)
        return h, coords_curr

class DeepKernelNN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1):
        super(DeepKernelNN, self).__init__()
        self.depth = depth
        self.fc1 = torch.nn.Linear(in_width, width)
        # width is the hidden dim
        self.num_layers = depth

        self.fc2 = torch.nn.Linear(width, 1)
        
        self.processor = nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'
        #self.processor.append(self.conv1)
        
        processor_layer=self.build_processor_model()
        for _ in range(self.num_layers):
            kernel = DenseNet([ker_in, ker_width//2, ker_width, width**2], torch.nn.ReLU)
            self.processor.append(processor_layer(width, width, kernel, aggr='mean'))

    def build_processor_model(self):
        return NNConv_old
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        #print('x shape {}'.format(x.shape))
        for k in range(self.depth):
            #print('x shape {} at depth {}'.format(x.shape, k))
            x = F.relu(self.processor[k](x, edge_index, edge_attr))
            
        x = self.fc2(x)
        return x