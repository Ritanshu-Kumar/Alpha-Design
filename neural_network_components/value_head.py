import torch
import torch.nn as nn

#so here we are gonna define the neural network arch for the value head

# so it goes eomthing similar to the policy head: 
# d_size(p_size)->d_size*4->d_size*2->1

# and then the activations would be relu(for now, for simplicity, considering gelu and silu too), and all of these are fully connected layers(dense)

#same as in policy, head missed gelu->relu, layer normalization and dropout, so adding it here
#and also reversing the mechnism of ffn by reducing the fetures by 2, and getting to 1, as you boil directly to 1, obviously it would fuck up the training
#learned it over here 🤓

class ValueHead(nn.Module):
    # def __init__(self, p_size):
    #     super(ValueHead, self).__init__()
    #     self.fc1 = nn.Linear(p_size, p_size*4)
    #     self.fc2 = nn.Linear(p_size*4, p_size*2)
    #     self.fc3 = nn.Linear(p_size*2, 1)  
    #     self.relu = nn.ReLU()

    # def forward(self, x):
    #     x = self.relu(self.fc1(x))    
    #     x = self.relu(self.fc2(x))    
    #     x = self.fc3(x)              
    #     return x
    def __init__(self, feature_dim):
        super(ValueHead, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.LayerNorm(),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(feature_dim//2, feature_dim//4),
            nn.LayerNorm(),
            nn.GELU(),
            nn.Dropout(0.05),

            nn.Linear(feature_dim//4, 1)
        )
    
    def forward(self, x):
        return self.layers(x)