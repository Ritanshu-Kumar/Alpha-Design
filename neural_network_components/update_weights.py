import torch
import torch.nn as nn
import torch.optim as optim

#so in this file we are specifically assigning the update weights for the neural networks we have, and then using the selcted optim

def update_weights(model, loss, optimizer):
    optimizer.zero_grad()  #clearing the previous computed gradients
    loss.backward()  #fresh gradient calculation
    optimizer.step() #updating weights based on the loss gradients
    return model