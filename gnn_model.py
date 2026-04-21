import torch
from torch import nn
from torch_geometric.nn.conv import NNConv, GATConv
from torch_geometric.nn import MLP, global_add_pool
from torch.nn import GRUCell, LSTMCell



class corr_gnn (nn.Module):

    def __init__ (
            self, 
            in_dimension: int,
            hidden_dimension: int,
            out_dimension: int = 1,
            T:int = 5,
            version: str = "v1"
    ):
        super().__init__()
        self.in_dimension = in_dimension
        self.hidden_dimension = hidden_dimension
        self.out_dimension = out_dimension
        self.T = T
        self.edge_attribute = 5
        if version == "v1" or version == "v2" or version == "v3":
            self.version = version
        else:
            raise ValueError(f'Version {version} is not implemented. Please choose between v1 and v2')

        # Define initial projection layer
        self.projection_layer = MLP([self.in_dimension, self.hidden_dimension])
        self.node_normalization = nn.LayerNorm(self.hidden_dimension)
        if self.version == "v1" or self.version == "v2":

            # An NNConv is defined for the MPNN convulution
            self.convolution = NNConv(in_channels=self.hidden_dimension, out_channels=self.hidden_dimension,
                                    nn=MLP([self.edge_attribute, 2 * self.hidden_dimension,
                                            self.hidden_dimension * self.hidden_dimension]),
                                    aggr='add')

            # Define the node update function
            if self.version == "v1":
                self.node_update = GRUCell(input_size=self.hidden_dimension,hidden_size=self.hidden_dimension)
            elif self.version == "v2":
                self.node_update = LSTMCell(input_size=self.hidden_dimension, hidden_size=self.hidden_dimension)

            # Define normalization after node update
            self.norm = nn.LayerNorm(self.hidden_dimension)

            # Define the neural networks that will be used during readout
            self.mlp_hidden_and_initial_state = MLP([2 * self.hidden_dimension, self.hidden_dimension * 4, hidden_dimension])
            self.mlp_hidden_state = MLP([self.hidden_dimension, 2 * self.hidden_dimension, self.hidden_dimension])

        else:

            # For v3, an architecture based on GATConv is built

            # Attention NNs used for message passing
            self.initial_convolution = GATConv(in_channels = hidden_dimension, out_channels= hidden_dimension,
                                               heads=4, concat = True, edge_dim=self.edge_attribute)
            self.intermediary_convolution = GATConv(in_channels = 4 * hidden_dimension, out_channels= hidden_dimension,
                                                    heads = 4, concat = True, edge_dim=self.edge_attribute)
            self.message_passing_norm = nn.LayerNorm(4 * self.hidden_dimension)
            
            # NN used for readout
            self.readout_convolution = GATConv(in_channels=4 * hidden_dimension, out_channels= hidden_dimension,
                                               heads = 1, concat = False, edge_dim=self.edge_attribute)

        # Define output layer
        self.mlp_output = MLP([hidden_dimension, hidden_dimension * 2, out_dimension])

        # Define graph normalization layer
        self.graph_norm = nn.LayerNorm(self.hidden_dimension)


    def forward(self, node_feature_matrix, edge_indices_matrix, edge_feature_matrix, batch):

        # The nodes are embedded into the hidden dimension and normalizes
        h = self.projection_layer(node_feature_matrix)
        h = self.node_normalization(h)
        h_0 = h.clone()

        if self.version == "v1" or self.version == "v2":
            # Cell state vector needed for LSTM
            if self.version == "v2":
                c = torch.zeros_like(h)

            # Message passing takes place
            for _ in range(self.T):
                m_timestep = self.convolution(h, edge_indices_matrix, edge_feature_matrix)
                if self.version == "v1":
                    h = self.node_update(m_timestep, h)
                elif self.version == "v2":
                    h, c = self.node_update(m_timestep, (h, c)) # The LSTM outputs both the hidden and cell states
                h = self.norm(h) # Normalization

            # Readout takes place
            i = torch.sigmoid(self.mlp_hidden_and_initial_state(torch.cat([h, h_0], dim=-1)))
            j = self.mlp_hidden_state(h)
            node_scores = i * j

        else:

            # Message aggregation takes place
            for timestep in range(self.T):
                if timestep == 0:
                    h_new = self.initial_convolution(h, edge_indices_matrix, edge_feature_matrix)
                    h = self.message_passing_norm(h_new) # Skip connections not yet applied due to different dimensionality
                else:
                    h_new = self.intermediary_convolution(h, edge_indices_matrix, edge_feature_matrix)
                    h = self.message_passing_norm(h_new + h) # Implements skip connection with normalization

            # Readout takes place
            node_scores = self.readout_convolution(h, edge_indices_matrix, edge_feature_matrix)
            node_scores = self.node_normalization(node_scores + h_0) # Skip connection with initial node state

        graph_embedding = global_add_pool(node_scores, batch)
        #graph_embedding = self.graph_norm(graph_embedding) # Normalization layer
        output = self.mlp_output(graph_embedding)
    
        return output