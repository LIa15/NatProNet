import torch.nn as nn
import numpy as np
from typing import Optional
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear, Parameter
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax
import torch
from torch_geometric.nn import MessagePassing
import esm
# from torch_geometric.nn.models import AttentiveFP


device = torch.device('cuda')
esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()


# The GATEConv and AttentiveFP layers are taken from the PyTorch Geometric (PyG) version 2.7.0 documentation.
# Since my experimental environment uses a lower version of PyTorch, I was concerned about potential compatibility
# issues with newer versions of PyG. Therefore, I ported the source code from the higher version to my setup.
# In the latest version of PyG, you can directly
# import AttentiveFP from torch_geometric.nn.models.
# The Attentive FP model for molecular representation learning from the
#     `"Pushing the Boundaries of Molecular Representation for Drug Discovery
#     with the Graph Attention Mechanism"
#     <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
#     graph attention mechanisms.


class GATEConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out + self.bias
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j @ self.att_l.t()).squeeze(-1)
        alpha_i = (x_i @ self.att_r.t()).squeeze(-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return self.lin2(x_j) * alpha.unsqueeze(-1)


class AttentiveFP(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                batch: Tensor) -> Tensor:
        """"""
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))

        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = F.elu_(conv(x, edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu_()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)

    def jittable(self) -> 'AttentiveFP':
        self.gate_conv = self.gate_conv.jittable()
        self.atom_convs = torch.nn.ModuleList(
            [conv.jittable() for conv in self.atom_convs])
        self.mol_conv = self.mol_conv.jittable()
        return self

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers}, '
                f'num_timesteps={self.num_timesteps}'
                f')')


def GNN(layers):
    return AttentiveFP(in_channels=39,
                                  hidden_channels=167,
                                  out_channels=167,
                                  edge_dim=10,
                                  num_layers=layers,
                                  num_timesteps=2,
                                  dropout=0.1
                                  )


class Classifier(nn.Module):
    def __init__(self, compoundDim, proteinDim, hiddenDim, outDim):
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = compoundDim + proteinDim
        for dim in hiddenDim:
            self.layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        self.layers.append(nn.Linear(prev_dim, outDim))

    def forward(self, compound_feature, protein_feature):
        # Assume that the compound and protein have already been properly processed and can be directly concatenated
        x = torch.cat((compound_feature, protein_feature), dim=1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x



class FlexibleNNClassifier(nn.Module):
    def __init__(self, compoundNN, classifier):
        super().__init__()
        self.compoundNN = compoundNN
        self.esmmodel = esm_model
        self.classifier = classifier
        self.lin = nn.Linear(640, 512)
        self.attention = nn.Linear(512, 1)


    def forward(self, inputs, proteins):
        compound_x, compound_edge_index, compound_edge_attr = inputs.x, inputs.edge_index, inputs.edge_attr
        compound_feature = self.compoundNN(compound_x, compound_edge_index, compound_edge_attr, inputs.batch)

        with torch.no_grad():
            results = self.esmmodel(proteins, repr_layers=[30])
        token_representations = results["representations"][30]
        protein_vector = token_representations[:, 1:, :]
        protein_vector = self.lin(torch.squeeze(protein_vector, 1))
        weights = torch.softmax(self.attention(protein_vector), dim=1)  # [batch_size, len, 1]
        protein_vector = torch.sum(weights * protein_vector, dim=1)  # [batch_size, 512]

        out = self.classifier(compound_feature, protein_vector)
        return out

    def __call__(self, data, proteins, train=True):
        correct_interaction = data.y
        predicted_interaction = self.forward(data, proteins)
        if train:
            criterion = torch.nn.CrossEntropyLoss().to(device)
            loss = criterion(predicted_interaction, correct_interaction)
            return loss, predicted_interaction
        else:
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            return correct_labels, predicted_labels, predicted_scores
