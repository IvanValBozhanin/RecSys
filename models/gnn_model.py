import torch.nn as nn
import Modules.architectures as archit
import Utils.graphML as gml



class MovieLensGNN(nn.Module):


# TODO: set grid for stochastic search for hyperparameters



    def __init__(self, cov_matrix, input_dim):
        super(MovieLensGNN, self).__init__()
        dimNodeSignals = [1, 16, 1]                     # TODO: for search, try multiple diff values and size.
        nFilterTaps = [2] * (len(dimNodeSignals) - 1) # values of each element >= 2|||| TODO: [2]/[3] for search.


        self.gnn = archit.SelectionGNN(dimNodeSignals, # list: 1st element is the #Feature; after that - hidden dimentions of each filter; last:  output size (1 rating per user per movie)
                                       nFilterTaps, # number of k per each layer
                                       True,
                                       nn.ReLU,
                                       [input_dim] * len(nFilterTaps), # input_dim is the number of features
                                       gml.NoPool, # pooling layer not needed for now.
                                       [1, 1],
                                       [1], #layers and number of neurons in each layer;;; CHECK IF len == 1 ||| first element here is the last element in the dimNodeSignals.
                                       # TODO: MLP input size as output of dimNodeSignals and try with different hidden layers (2-3) and aoutput is always 1..
                                       cov_matrix)

    def forward(self, x):
        return self.gnn(x)
