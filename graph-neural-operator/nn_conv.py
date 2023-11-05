import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset, uniform
from torch.nn import Sequential, Linear, ReLU, LayerNorm
import torch_scatter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################## NKN ###################################
class NNConv_NKN(MessagePassing):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='add',
                 root_weight=False,
                 bias=True,
                 **kwargs):
        super(NNConv_NKN, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr
        self.eigen_mat = torch.Tensor(256*out_channels,256*out_channels)

        if root_weight:
            self.root = Parameter(torch.rand(in_channels, out_channels))
            #self.root2 = Parameter(torch.rand((in_channels, out_channels)))
            #self.root3 = Parameter(torch.rand((in_channels, out_channels)))
            #print(self.root3)
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        size = self.in_channels
        uniform(size, self.root)
        uniform(size, self.bias)

    def calc_eigen_mat(self,edge_index,edge_attr):
        #x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        #weight = torch.ones(4232, 1, 1)
        self.eigen_mat = torch.zeros(256 * self.in_channels, 256 * self.out_channels)
        j = 0
        length_index = torch.zeros(256)
        #print("assemble matrix: ")
        for i in range(256):
            j_last = j
            while j < 2116 and edge_index[0, j].data <= i + 1e-10:
                self.eigen_mat[i * self.in_channels: (i + 1) * self.in_channels, (edge_index[1, j]) * self.in_channels:(edge_index[1, j] + 1) * self.in_channels] += weight[j, :, :]#torch.mm(self.root3, self.root2)
                self.eigen_mat[i * self.in_channels: (i + 1) * self.in_channels,i* self.in_channels:(i + 1) * self.in_channels] -= weight[j, :, :]
                #self.eigen_mat[i * self.in_channels: (i + 1) * self.in_channels,i * self.in_channels:(i+ 1) * self.in_channels] -= torch.mm(self.root3,self.root2)
                #self.eigen_mat[(i + 256) * 64:(i + 256 + 1) * 64,(edge_index[1, j] + 256) * 64:(edge_index[1, j] + 1 + 256) * 64] += weight[j + 2116, :, :]
                j = j + 1
            length_index[i] = j - j_last
        for i in range(256):
            self.eigen_mat[:, i * self.in_channels:(i + 1) * self.in_channels] /= length_index[i]
            #self.eigen_mat[:, (i + 256) * 64:(i + 256 + 1) * 64] /= length_index[i]
            # self.ave_mat[i, :, :] /= (j-j_last)
            # self.ave_mat[i+256, :, :] /= (j - j_last)
            # print(j)
        #print("Add diagonal term: ")
        #for i in range(256):
        #    self.eigen_mat[i * 64:(i + 1) * 64, i * 64:(i + 1) * 64] = self.root
            #self.eigen_mat[i*self.in_channels:(i+1)*self.in_channels, i*self.in_channels:(i+1)*self.in_channels] += torch.eye(self.in_channels)
            #self.eigen_mat[(i+256) * 64:(i + 1+256) * 64, (i+1) * 64:(i + 1 + 256) * 64] += self.root



    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, x_i,pseudo,edge_index):
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        #weight = torch.eye(4232,1,1)
        #print(x_j.shape)
        #print(self.root3)
        #cross_prod = torch.matmul(x_j.unsqueeze(1), x_i.unsqueeze(2)).squeeze(1)
        edge_mat = torch.matmul((x_j-x_i).unsqueeze(1), weight).squeeze(1)
        #print(edge_mat)
        """
        self.ave_mat = torch.zeros(512*self.in_channels,512*self.out_channels)
        j = 0
        length_index = torch.zeros(256)
        for i in range(256):
            j_last = j
            while j < 2116 and edge_index[0,j].data <= i + 1e-10:
                self.ave_mat[i*64: (i+1)*64, (edge_index[1, j])*64:(edge_index[1, j]+1)*64] += weight[j, :, :]
                self.ave_mat[(i+256)*64:(i+256+1)*64, (edge_index[1, j]+256)*64:(edge_index[1, j]+1+256)*64] += weight[j+2116, :, :]
                j = j+1
            length_index[i] = j-j_last
        for i in range(256):
            self.ave_mat[:,i*64:(i+1)*64] /= length_index[i]
            self.ave_mat[:,(i+256)*64:(i+256+1)*64] /= length_index[i]
            #self.ave_mat[i, :, :] /= (j-j_last)
            #self.ave_mat[i+256, :, :] /= (j - j_last)
            #print(j)
        """
        return edge_mat

    def update(self, aggr_out, x):
        if self.root is not None:
            #aggr_out = torch.mm(aggr_out, self.root2) + torch.mm(x, self.root)
            #aggr_out = torch.mm(aggr_out, self.root2) + x
            aggr_out = aggr_out + torch.mm(x, self.root)
            #aggr_out = aggr_out + x
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

########################## MGN ###################################

class ProcessorLayer(MessagePassing):
    """ The Processor layer in the MGN model. This layer is used to update the node embeddings and edge embeddings.
    """

    def __init__(self, in_channels, out_channels,  **kwargs):
        super(ProcessorLayer, self).__init__(  **kwargs )
        """
        in_channels: dim of node embeddings [128], out_channels: dim of edge embeddings [128]

        """

        # Note that the node and edge encoders both have the same hidden dimension
        # size. This means that the input of the edge processor will always be
        # three times the specified hidden dimension
        # (input: adjacent node embeddings and self embeddings)
        self.edge_mlp = Sequential(Linear( 3* in_channels , out_channels),
                                   ReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))

        self.node_mlp = Sequential(Linear( 2* in_channels , out_channels),
                                   ReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))


        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr, size = None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shpae [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]

        """

        out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = size) # out has the shape of [E, out_channels]

        updated_nodes = torch.cat([x,out],dim=1)        # Complete the aggregation through self-aggregation

        updated_nodes = x + self.node_mlp(updated_nodes) # residual connection

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        source_node: x_i has the shape of [E, in_channels]
        target_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]

        The messages that are passed are the raw embeddings. These are not processed.
        """

        updated_edges=torch.cat([x_i, x_j, edge_attr], dim = 1) # tmp_emb has the shape of [E, 3 * in_channels]
        updated_edges=self.edge_mlp(updated_edges)+edge_attr

        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size = None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """

        # The axis along which to index number of nodes.
        node_dim = 0
        out = torch_scatter.scatter(updated_edges, 
                                    edge_index[0, :],
                                    dim=node_dim, 
                                    reduce = self.aggr)

        return out, updated_edges

########################## MGO (updated) ###################################
class OperatorProcessorLayer(MessagePassing):
    """ The Processor layer in the MGN model. This layer is used to update the node embeddings and edge embeddings.
    """
    def __init__(self, in_channels, out_channels, nn, aggr='add', **kwargs):
        super(OperatorProcessorLayer, self).__init__(  **kwargs )
        """
        in_channels: dim of node embeddings [128], out_channels: dim of edge embeddings [128]

        """
        self.nn = nn
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr = aggr
        
        self.edge_mlp = Sequential(Linear( 3* in_channels , out_channels),
                                   ReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))

        self.node_mlp = Sequential(Linear( 2* in_channels , out_channels),
                                   ReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # Reset parameters of the neural network for edge weighting
        reset(self.nn)
        # Reset parameters of the node MLP
        self.node_mlp.apply(self.weights_init)
        self.edge_mlp.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, Linear):
            uniform(m.weight.size(0), m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr, size = None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shpae [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]

        """
        out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = size) # out has the shape of [E, out_channels]
        updated_nodes = torch.cat([x,out],dim=1)        # Complete the aggregation through self-aggregation
        updated_nodes = x + self.node_mlp(updated_nodes) # residual connection

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        source_node: x_i has the shape of [E, in_channels]
        target_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]

        The messages that are passed are the raw embeddings. These are not processed.
        """
        updated_edges=torch.cat([x_i, x_j, edge_attr], dim = 1) # tmp_emb has the shape of [E, 3 * in_channels]
        updated_edges=self.edge_mlp(updated_edges)+edge_attr
        #print("update_edges shape {}".format(updated_edges.shape))
        
        weight = self.nn(updated_edges).view(-1, self.in_channels, self.out_channels)
        #print("weight shape {}".format(weight.shape))
        # Update the node features using the computed weights
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1), updated_edges

    def aggregate(self, inputs, edge_index, dim_size = None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """        
        weighted_edges, updated_edges = inputs[0], inputs[1]
        # The axis along which to index number of nodes.
        node_dim = 0
        out = torch_scatter.scatter(weighted_edges, 
                                    edge_index[0, :],
                                    dim=node_dim, 
                                    reduce = self.aggr)

        return out, updated_edges

########################## MGO (Deprecated) ###################################
class ProcessorOperatorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, nn, aggr='add', **kwargs):
        super(ProcessorOperatorLayer, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn

        # This MLP will process the node features after message passing
        self.node_mlp = Sequential(
            Linear(2 * in_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
            LayerNorm(out_channels)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        # Reset parameters of the neural network for edge weighting
        reset(self.nn)
        # Reset parameters of the node MLP
        self.node_mlp.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, Linear):
            uniform(m.weight.size(0), m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        # Propagate messages using the edge indices and the dynamically computed edge weights
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Post-processing of node features after message passing
        updated_nodes = torch.cat([x, out], dim=1)
        updated_nodes = x + self.node_mlp(updated_nodes)  # Residual connection

        return updated_nodes

    def message(self, x_j, edge_attr):
        # Compute the edge weights using the neural network
        weight = self.nn(edge_attr).view(-1, self.in_channels, self.out_channels)

        # Update the node features using the computed weights
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def aggregate(self, inputs, index, dim_size=None):
        # The aggregation function as specified by 'self.aggr', e.g., sum, mean, or max.
        node_dim = self.node_dim
        return torch_scatter.scatter(inputs, index, dim=node_dim, dim_size=dim_size, reduce=self.aggr)

    def __repr__(self):
        return '{}(in_channels={}, out_channels={})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class NNConv(MessagePassing):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NNConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        weight_diag = torch.diag_embed(self.nn(pseudo)).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight_diag).squeeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class NNConv_Gaussian(MessagePassing):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NNConv_Gaussian, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        one = torch.ones(1).to(device)
        a = 1 / torch.sqrt(torch.abs(pseudo[:,1] * pseudo[:,2]))
        # print('a',torch.isnan(a))
        b = torch.exp(-1 * (pseudo[:, 0] ** 2).view(-1, 1) / (self.nn(one) ** 2).view(1, -1))
        # print('b',torch.isnan(b))
        weight_guass = a.reshape(-1,1).repeat(1,64) * b
        # print('w',torch.isnan(weight_guass))
        weight_guass = torch.diag_embed(weight_guass).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight_guass).squeeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class NNConv_old(MessagePassing):
    r"""The continuous kernel-based convolutional operator from the
    `"Neural Message Passing for Quantum Chemistry"
    <https://arxiv.org/abs/1704.01212>`_ paper.
    This convolution is also known as the edge-conditioned convolution from the
    `"Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on
    Graphs" <https://arxiv.org/abs/1704.02901>`_ paper (see
    :class:`torch_geometric.nn.conv.ECConv` for an alias):

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta} \mathbf{x}_i +
        \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \cdot
        h_{\mathbf{\Theta}}(\mathbf{e}_{i,j}),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.*
    a MLP.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps edge features :obj:`edge_attr` of shape :obj:`[-1,
            num_edge_features]` to shape
            :obj:`[-1, in_channels * out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`.
        aggr (string, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add the transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 aggr='add',
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NNConv_old, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nn = nn
        self.aggr = aggr

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        size = self.in_channels
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        #print('pseudo (edge attribute {})'.format(pseudo.shape))
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        #print('x_j shape {}'.format(x_j.shape))
        #print('weight shape {}'.format(weight.shape))
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

ECConv = NNConv
