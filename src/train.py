"""
@author : Ian Huang
The training script for the saliency and non-saliency models
"""
import torch
from torch.utils.data import Dataloader
import numpy as np
import os
from dataloader import SalconDataset
from models.fcn16s import FCN16s
from utils import loss

def trainNSNet():
    """ Method to train the network without saliency
    """
    # TODO: running training on GPU?

    # TODO: relevant transformations?
    # define dataset and dataloader
    sal_data= SalconDataset('../data/salcon_data/train_rgb_gaze_sem.csv',
                            '../data/salcon_data/semid.csv',
                            train=True, classes=['chair', 'dining_table'],
                            verbose=True)
    sal_loader = Dataloader(sal_data, batch_size=FLAGS.batch_size, shuffle=True, 
                            num_workers=FLAGS.num_cpus)
    # define the model
    model = FCN16s(n_class = FLAGS.num_classes)

    # TODO: define loss function and optimizer
    criterion = loss.cross_entropy2d
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    # define training sequence
    for epoch in range(FLAGS.num_epochs):
        running_loss = 0.0
        for i, data in enumerate(sal_loader, 0):
            rgb, gaze, sem = data
            
            optimizer.zero_grad()
            # every 200 or so iterations, save the model and display results
            outputs = model(rgb)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0 
                # TODO: save model
                # TODO: validation?
                # TODO: plot out the losses in tensorboard?

def trainSNet():
    """ Method to train the network without saliency
    """
    # define dataset and dataloader 
    # define loss function
    # define the model
    # define train sequence
        # every 200 or so iterations, save the model and display results.

if __name__=='__main__':

    # parse args
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        type=int,
        default = 16,
        help="""Batch size for training"""
    ) 
    parser.add_argument(
        '--num_cpus',
        type=int,
        default =4,
        help="""Number of cpus for dataloader"""
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default = 50,
        help="""Number of epochs for training"""
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default = 3,
        help="""Number of epochs for training"""
    )
    
    FLAGS, unparsed = parser.parse_known_args()
   
    
    

    

    

    



