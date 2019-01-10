import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import numpy as np

class Convnet(nn.Module):
    def __init__(self):
        # define the model
        super(Convnet, self).__init__()
        self.conv1 = nn.conv2d(1,6,5) 
    
    def forward(self, x):
        pass


class SalConvNet:
    def __init__(self):
        pass

if __name__=='__main__':
   NSnet = Convnet()




