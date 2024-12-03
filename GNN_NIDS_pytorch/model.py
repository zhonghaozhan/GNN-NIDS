import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class GNNLayer(MessagePassing):
    def __init__(self, node_state_dim, dropout_rate=0.5):
        super(GNNLayer, self).__init__(aggr='mean')
        self.node_state_dim = node_state_dim
        
        # Message functions with layer normalization
        self.message_nn = nn.Sequential(
            nn.Linear(node_state_dim * 2, node_state_dim),
            nn.LayerNorm(node_state_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Update function
        self.update_nn = nn.Sequential(
            nn.Linear(node_state_dim * 2, node_state_dim),
            nn.LayerNorm(node_state_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, edge_index):
        # Add self-loops to the adjacency matrix
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Start propagating messages
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # Construct messages from source nodes x_j to target nodes x_i
        tmp = torch.cat([x_i, x_j], dim=1)
        return self.message_nn(tmp)

    def update(self, aggr_out, x):
        # Combine aggregated messages with current state
        tmp = torch.cat([aggr_out, x], dim=1)
        return self.update_nn(tmp)

class GNN(nn.Module):
    def __init__(self, config):
        super(GNN, self).__init__()
        self.config = config
        self.use_checkpointing = False
        
        self.input_dim = config.INPUT_DIM
        self.hidden_dim = config.NODE_STATE_DIM
        self.output_dim = config.OUTPUT_DIM
        self.num_layers = config.NUM_LAYERS
        self.dropout_rate = config.DROPOUT
        
        # Initial feature transformation
        self.input_transform = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate)
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.dropout_rate))
        
        # Output transformation
        self.output_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def enable_checkpointing(self):
        self.use_checkpointing = True
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Sample edges if configured
        if hasattr(self.config, 'EDGE_SAMPLING_RATIO') and self.config.EDGE_SAMPLING_RATIO < 1.0 and self.training:
            num_edges = edge_index.size(1)
            perm = torch.randperm(num_edges, device=edge_index.device)
            num_sampled = int(num_edges * self.config.EDGE_SAMPLING_RATIO)
            edge_index = edge_index[:, perm[:num_sampled]]
        
        # Initial feature transformation
        x = self.input_transform(x)
        
        # Apply GNN layers with residual connections
        for i, gnn in enumerate(self.gnn_layers):
            identity = x
            
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(gnn, x, edge_index)
            else:
                x = gnn(x, edge_index)
            
            # Residual connection with scaling
            if i > 0:  # Skip first layer as dimensions might not match
                x = 0.5 * (x + identity)  # Scale the residual connection
        
        # Final classification
        x = self.output_transform(x)
        return F.log_softmax(x, dim=-1)  # Use log_softmax for numerical stability
