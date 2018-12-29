"""
@author : Ian Huang
The dataloader for the Salcon dataset
"""
import os 
from scipy.misc import imread
import numpy as np
import pandas as pd
import pickle # for loading semantic labels

class SalconDataloader(object):
    def __init__(self, data_path_csv, sem_id_csv, gt_type='pix', 
                 random_draw=True, classes=[]):
        """ Constructor for the Salcon dataset
        Input:
            rgb_csv (str): path to rgb data csv
            sal_csv (str): path to saliency map csv
            sem_csv (str): path to semantic labeling csv
            sem_id_csv (str): path to semantic metadata csv
            gt_type (str): ground truth type
            random_draw (optional, bool): true if the draw of every sample is
                random, false if it goes down the list.
            classes (optional, list): list of classes that will be extracted
                from the dataset. 
        """
        self.data_path_csv = data_path_csv
        self.sem_id_csv = sem_id_csv
        self.gt_type = gt_type
        assert self.gt_type == 'pix' or self.gt_type == 'bbox'
        
        self.random_draw = random_draw
        self.classes = classes
        # making label2id sem2id 
        label2id_pd = pd.read_csv(self.sem_id_csv)
        sem2id = {}
        for idx, element in label2id_pd.iterrows():
            sem2id[element['name']] = element['id']
        assert all([element in sem2id for element in self.classes]),\
                'Elements of self.classes needs to be {}'.\
                format(list(sem2id.values()))
        # conversion from self.classes to id's
        self.focus_ids = set([sem2id[element] for element in self.classes])
        if len(self.focus_ids) == 0: # if empty:
            self.focus_ids = set(list(sem2id.values())) # include everything
        
        # reading csv
        self.data_path = pd.read_csv(self.data_path_csv)
        # preprocess the classes to hold only required classes
        self.available_indices = self.filter_by_classes()
   
    def get_label2class_conv(self, id_num):
        pass 
    def get_class2label_conv(self, class_str):
        pass

    def set_desired_labels(self):
        pass
    
    def filter_by_classes(self):
        """ find images with at least one instance self.focus_ids in sem labels
        Returns:
            indices (list): a list of row indices in self.data_path where 
                            conditions hold.
        """
        indices = [] 
        for index, row in self.data_path.iterrows():
            path = row['semantic_label']
            assert os.path.exists(path), \
                'sematic_label path {} does not exist'.format(path)
            assert '.pkl' in path, 'Expected {} to be a pkl file'.format(path)
            
            with open(path, 'rb') as f:
                labels = pickle.load(f)
                if any([element in labels.flatten() for element in \
                        list(self.focus_ids)]):
                    # save index
                    indices.append(index)
        return indices 

    def __iter__(self):
        """ gets a random sample if self.random=True, or ordered down the list 
            if False
        """
        self.read_index = 0 
        return self

    def __next__(self):
        """ gets a random sample if self.random=True, or ordered down the list 
            if False
        """
        if self.random_draw:
            # reset read_index as a random number in [0, length-1]
            self.read_index = random.rand_int(0, len(self.available_indices)-1)
            prod = self.data_path.iloc(self.available_indices[self.read_index])
        else:
            #increment read_index  
            prod = self.data_path.iloc(self.available_indices[self.read_index])
            self.read_index += 1
        
        rgb = imread(prod['rgb'])
        gaze = imread(prod['gaze'])
        with open(prod['semantic_label'], 'rb') as f:
            sem_label = pickle.load(f)
        
        # zeroing out the classes not in the selected classes 
        sem_label_masked = np.zeros_like(sem_label)
        for cat in list(self.focus_ids):
            vert, horiz = np.where(sem_label == cat)
            sem_label_masked[vert, horiz] = cat
        
        if self.gt_type == 'bbox':
            # TODO: yet to be implemented
            pass
    
        return (rgb, gaze, target)



if __name__=='__main__':
    saldata = SalconDataloader('../data/train_rgb_gaze_sem.csv', '../data/semid.csv')
