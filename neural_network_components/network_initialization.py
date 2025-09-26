import torch
import torch.nn as nn
from .forward_pass import NeuralNetworkForwardPass

class NetworkInitializer:

    # @staticmethod
    # def _apply_he_initialization(network):
    #     for module in network.modules():
    #         if isinstance(module, nn.Linear):
    #             nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
    #             if module.bias is not None:
    #                 nn.init.zeros_(module.bias)

    def __appy_enahanced_initialization(network):
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.weight)
                nn.init.ones_(module.bias)
    
    @staticmethod
    def setup_network(design_param_count, device="cpu", hidden_dim=256, depth=3):

        adaptive_hidden_dim = min(512, max(hidden_dim, design_param_count*4))
        network = NeuralNetworkForwardPass(
            design_param_count, 
            hidden_dim=adaptive_hidden_dim,
            depth=depth,
            dropout=0.1
        )
        network.to(device)
        
        NetworkInitializer.__appy_enahanced_initialization(network)

        param_count = sum(p.numel() for p in network.parameters() if p.requires_grad)

        return network, param_count
    
    @staticmethod
    def get_network_info(network):
        total_params = sum(p.numel() for p in network.parameters())
        trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'feature_trunk_params': sum(p.numel() for p in network.feature_trunk.parameters()),
            'policy_head_params': sum(p.numel() for p in network.policy_head.parameters()),
            'value_head_params': sum(p.numel() for p in network.value_head.parameters())
        }