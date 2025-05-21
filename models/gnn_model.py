import numpy as np
import torch
import torch.nn as nn
import Modules.architectures as archit
import Utils.graphML as gml
from typing import List

class MovieLensGNN(nn.Module):


# TODO: set grid for stochastic search for hyperparameters



    def __init__(
            self,
            cov_matrix: torch.Tensor,
            num_movies: int,
            dimNodeSignals: List[int],
            nTaps: int,
            dimLayersMLP: List[int],
            num_users: int
    ):
        assert num_users > 0, "MOVIELENSGNN: num_users must be greater than 0"
        assert dimNodeSignals[0] == num_movies, "MOVIELENSGNN: dimNodeSignals[0] must be equal to input_feature_dim_per_node"

        super(MovieLensGNN, self).__init__()
        nFilterTaps_list = [nTaps] * (len(dimNodeSignals) - 1) # values of each element >= 2|||| TODO: [2]/[3] for search.
        nSelectedNodes_list = [num_users] * len(nFilterTaps_list)

        self.gnn_layers = archit.SelectionGNN(
            dimNodeSignals,  # [num_movies, hidden1, gnn_output_feat_dim]
            nFilterTaps_list,
            True,  # bias
            nn.LeakyReLU,
            nSelectedNodes_list,
            gml.NoPool,
            [1] * len(nFilterTaps_list),
            [],  # NO MLP INSIDE SelectionGNN for this structure. We will add our own MLP head for predictions
            cov_matrix,  # User-User GSO
            average=False  # We want per-node features from GNN
        )

        gnn_output_feat_dim = dimNodeSignals[-1]

        mlp_layers = []
        if dimLayersMLP and len(dimLayersMLP) > 0:
            if dimLayersMLP[0] != gnn_output_feat_dim:
                raise ValueError("MOVIELENSGNN: dimLayersMLP[0] must be equal to gnn_output_feat_dim")
            for i in range(len(dimLayersMLP) - 1):
                mlp_layers.append(nn.Linear(dimLayersMLP[i], dimLayersMLP[i + 1]))
                if i < len(dimLayersMLP) - 2:
                    mlp_layers.append(nn.LeakyReLU())

            if dimLayersMLP[-1] != num_movies:
                raise ValueError("MOVIELENSGNN: dimLayersMLP[-1] must be equal to num_movies")

        else:
            if gnn_output_feat_dim != num_movies:
                mlp_layers.append(nn.Linear(gnn_output_feat_dim, num_movies))

        self.prediction_mlp = nn.Sequential(*mlp_layers)


        # self.gnn = archit.SelectionGNN(
        #                             dimNodeSignals, # list: 1st element is the #Feature; after that - hidden dimentions of each filter; last:  output size (1 rating per user per movie)
        #                             nFilterTaps_list, # number of k per each layer
        #                             True,
        #                             nn.LeakyReLU, # todo: try leaky-relu.
        #                             nSelectedNodes_list, # input_dim is the number of features
        #                             gml.NoPool, # pooling layer not needed for now.
        #                             [1, 1],
        #                             dimLayersMLP, #layers and number of neurons in each layer;;; CHECK IF len == 1 ||| first element here is the last element in the dimNodeSignals.
        #                             # TODO: MLP input size as output of dimNodeSignals and try with different hidden layers (2-3) and output is always 1..
        #                             cov_matrix
        #                             )




    def forward(self, x_all_user_features):

        if x_all_user_features.ndim != 2 or \
                x_all_user_features.shape[1] != self.gnn_layers.F[0] or \
                x_all_user_features.shape[0] != self.gnn_layers.N[0]:  # N[0] is num_users from GSO
            raise ValueError(f"Input x_all_user_features has shape {x_all_user_features.shape}. "
                             f"Expected (num_users, num_movies_features) = "
                             f"({self.gnn_layers.N[0]}, {self.gnn_layers.F[0]})")

        gnn_input = x_all_user_features.permute(1, 0).unsqueeze(0)

        output_gnn_raw, _ = self.gnn_layers.splitForward(gnn_input)

        gnn_output_features = self.gnn_layers.GFL(gnn_input)

        user_embeddings_refined = gnn_output_features.squeeze(0).permute(1, 0)

        predicted_ratings = self.prediction_mlp(user_embeddings_refined)

        return predicted_ratings

