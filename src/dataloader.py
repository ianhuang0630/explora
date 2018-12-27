"""
@author : Ian Huang
The dataloader for the Salcon dataset
"""
import os 
from scipy.misc import imread
import numpy as np
import pickle # for loading semantic labels


class SalconDataloader(object):
    def __init__(self, rgb_csv, sal_csv, data_csv):
        pass
    
    def get_class_freq(self):
        pass
    
    def get_label2class_conv(self):
        pass

    def set_desired_labels(self):
        pass
    
    def __iter__(self, random=True):
        """ gets a random sample if random=True, or ordered down the list if False
        """
        pass

if __name__=='__main__':
    pass
