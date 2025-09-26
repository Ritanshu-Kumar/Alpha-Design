import torch
import torch.nn as nn

from .policy_head import PolicyHead
from .value_head import ValueHead

class NeuralNetworkForwardPass(nn.Module):
    def __init__(self, param_count, hidden_dim=256, depth=3, dropout=0.1):
        super(NeuralNetworkForwardPass, self).__init__()
        self.param_count = param_count
        #so here missed the shared feature thing, now learned a lesson from perplexity and adding it accordingly
        layers = []
        in_dim = param_count
        for i in range(depth):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim

        self.feature_trunk = nn.Sequential(*layers)

        self.policy_head = PolicyHead(hidden_dim, param_count)
        self.value_head = ValueHead(hidden_dim)

    def forward(self, design_params):
        features = self.feature_trunk(design_params)
        #so here we are getting the features from the trunk, and then passing it to the
        #policy and value heads to get the outputs
        policy_output = self.policy_head(features)
        value_output = self.value_head(features)
        return policy_output, value_output
    
    def predict_modifications(self, current_params):
        features = self.feature_trunk(current_params)
        return self.policy_head(features)
    
    def evaluate_design(self, current_params):
        features = self.feature_trunk(current_params)
        return self.value_head(features)