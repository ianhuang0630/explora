"""
@author : Ian Huang
The dataloader for the Salcon dataset
"""
import os 
from scipy.misc import imread
import numpy as np
import pandas as pd
import pickle # for loading semantic labels
from tqdm import tqdm
from multiprocessing import Pool
import random
from torch.utils.data import Dataset 

class SalconDataset(Dataset):
    """This is a Dataset object that needs to passed through Dataloader with
    an appropriately specified batchsize, shuffle/no shuffle ...etc
    """
    def __init__(self, data_path_csv, sem_id_csv, train=True, transform=None, 
                 gt_type='pix', classes=[], read_cache=True, verbose=True):
        """ Initializer for the Salcon dataset
        Input:
            data_path_csv (str): csv with path to training instances
            sem_id_csv (str): path to semantic metadata csv
            train (bool): True if dataloader is used for training
            gt_type (optional, str): ground truth type
            classes (optional, list): list of classes that will be extracted
                from the dataset.
            read_cache (optional, bool): True if the indices for a certain 
                set of classes is to be read from memory. False if the indices
                must be recomputed and resaved.
            verbose (optional, bool): True if verbose, false otherwise.
        """
        self.data_path_csv = data_path_csv
        self.sem_id_csv = sem_id_csv
        self.gt_type = gt_type
        assert self.gt_type == 'pix' or self.gt_type == 'bbox'
        self.verbose = verbose
        self.train = train
        self.read_cache = read_cache
        self.transform = transform

        self.classes = classes
        # making label2id sem2id 
        if self.verbose:
            print('reading semantic id mapping csv...')
        label2id_pd = pd.read_csv(self.sem_id_csv)
        self.sem2id = {}
        self.id2sem = {}
        if self.verbose:
            print('constructing semantic and id mappings...')
        for idx, element in label2id_pd.iterrows():
            self.sem2id[element['name']] = element['id']
            self.id2sem[element['id']] = element['name']
        assert all([element in self.sem2id for element in self.classes]),\
                'Elements of self.classes needs to be {}'.\
                format(list(self.sem2id.keys()))
        # conversion from self.classes to id's
        if self.verbose:
            print('getting focus classes ids...')
        self.focus_ids = set([self.sem2id[element] for element in self.classes])
        if len(self.focus_ids) == 0: # if empty:
            self.focus_ids = set(list(self.sem2id.values())) # include everything
        if self.verbose:
            print('loading data path csv...')
        # reading csv
        self.data_path = pd.read_csv(self.data_path_csv)
        # preprocess the classes to hold only required classes
        if self.verbose:
            print('Constructing list of indices in csv satisfying class requirements...')
        self.available_indices = self.filter_by_classes()
   
    def get_id2class_conv(self, id_num):
        return self.id2sem[id_num]

    def get_class2id_conv(self, class_str):
        return self.sem2id[class_str] 

    def filter_by_classes(self):
        """ find images with at least one instance self.focus_ids in sem labels
        Returns:
            indices (list): a list of row indices in self.data_path where 
                            conditions hold.
        """
        # if less than all categories are in target_id's:
        # creating name -- .[train/val]_[alphabetized classes]_indices.npy
        # under directory ../directory
        if self.train:
            filename = 'train_'+\
                        '+'.join(['_'.join(element.split(' ')) \
                                for element in sorted(self.classes)])+\
                        '_indices.npy'
        else:
            filename = 'val_'+\
                        '+'.join(['_'.join(element.split(' ')) \
                                for element in sorted(self.classes)])+\
                        '_indices.npy'

        save_path = os.path.join(os.path.dirname(self.data_path_csv),
                                filename)

        if len(self.focus_ids) < len(self.sem2id):
            if self.read_cache and os.path.exists(save_path):
                if self.verbose:
                    print('reading {}'.format(save_path)) 
                indices = np.load(save_path).tolist()
            else: # create and save indices 
                indices = [] 
                for index, row in tqdm(self.data_path.iterrows()):
                    path = row['semantic_label']
                    # some checks
                    assert os.path.exists(path), \
                        'semantic_label path {} does not exist'.format(path)
                    assert '.pkl' in path, 'Expected {} to be a pkl file'.\
                            format(path)
                    # checking if desired labels are in current dataset 
                    with open(path, 'rb') as f:
                        labels = pickle.load(f)
                        if any([element in labels.flatten() for element in \
                                list(self.focus_ids)]):
                            # save index
                            indices.append(index)
                # save indices at a given location
                if self.verbose:
                    print('saving to cache file {}'.format(save_path))
                np.save(save_path, np.array(indices))
                if self.verbose:
                    print('saved.')

        else: # all categories are kept, and no need to be saved
            indices = list(range(len(self.data_path)))
        return indices 
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):
        prod = self.data_path.iloc[idx]
        rgb = imread(element['rgb'])
        gaze = imread(element['gaze'])
        with open(element['semantic_label'], 'rb') as f:
            sem_label = pickle.load(f)
        
        # zeroing out the classes not in the selected classes 
        target = np.zeros_like(sem_label)
        for cat in list(self.focus_ids):
            vert, horiz = np.where(sem_label == cat)
            target[vert, horiz] = cat
        
        sample = {'rgb': rgb, 'gaze': gaze, 'target': target}
        if self.gt_type == 'bbox':
            # TODO: yet to be implemented
            raise ValueError('bbox gt not yet implemented')
        if self.transform:
            sample = self.transform(sample)
        return sample
        

class SalconDataloaderInHouse(object):
    """ Implementation of a dataloader that is Pytorch independent
    """
    def __init__(self, data_path_csv, sem_id_csv, gt_type='pix', 
                 random_draw=True, classes=[], train=True, read_cache=True, 
                 batch_size=1, verbose=True):
        """ Initializer for the Salcon dataset
        Input:
            data_path_csv (str): csv with path to training instances
            sem_id_csv (str): path to semantic metadata csv
            gt_type (optional, str): ground truth type
            random_draw (optional, bool): true if the draw of every sample is
                random, false if it goes down the list.
            classes (optional, list): list of classes that will be extracted
                from the dataset.
            train (optional, bool): True if dataloader is used for training
            read_cache (optional, bool): True if the indices for a certain 
                set of classes is to be read from memory. False if the indices
                must be recomputed and resaved.
            batch_size (optional, int): batch size
            verbose (optional, bool): True if verbose, false otherwise.
        """
        self.data_path_csv = data_path_csv
        self.sem_id_csv = sem_id_csv
        self.gt_type = gt_type
        assert self.gt_type == 'pix' or self.gt_type == 'bbox'
        self.verbose = verbose
        self.train = train
        self.read_cache = read_cache
        self.batch_size = batch_size 

        self.random_draw = random_draw
        if not train:
            assert not self.random_draw, 'random draw not supported for evaluation'

        self.classes = classes
        # making label2id sem2id 
        if self.verbose:
            print('reading semantic id mapping csv...')
        label2id_pd = pd.read_csv(self.sem_id_csv)
        self.sem2id = {}
        self.id2sem = {}
        if self.verbose:
            print('constructing semantic and id mappings...')
        for idx, element in label2id_pd.iterrows():
            self.sem2id[element['name']] = element['id']
            self.id2sem[element['id']] = element['name']
        assert all([element in self.sem2id for element in self.classes]),\
                'Elements of self.classes needs to be {}'.\
                format(list(self.sem2id.keys()))
        # conversion from self.classes to id's
        if self.verbose:
            print('getting focus classes ids...')
        self.focus_ids = set([self.sem2id[element] for element in self.classes])
        if len(self.focus_ids) == 0: # if empty:
            self.focus_ids = set(list(self.sem2id.values())) # include everything
        if self.verbose:
            print('loading data path csv...')
        # reading csv
        self.data_path = pd.read_csv(self.data_path_csv)
        # preprocess the classes to hold only required classes
        if self.verbose:
            print('Constructing list of indices in csv satisfying class requirements...')
        self.available_indices = self.filter_by_classes()
   
    def get_id2class_conv(self, id_num):
        return self.id2sem[id_num]

    def get_class2id_conv(self, class_str):
        return self.sem2id[class_str] 

    def filter_by_classes(self):
        """ find images with at least one instance self.focus_ids in sem labels
        Returns:
            indices (list): a list of row indices in self.data_path where 
                            conditions hold.
        """
        # if less than all categories are in target_id's:
        # creating name -- .[train/val]_[alphabetized classes]_indices.npy
        # under directory ../directory
        if self.train:
            filename = 'train_'+\
                        '+'.join(['_'.join(element.split(' ')) \
                                for element in sorted(self.classes)])+\
                        '_indices.npy'
        else:
            filename = 'val_'+\
                        '+'.join(['_'.join(element.split(' ')) \
                                for element in sorted(self.classes)])+\
                        '_indices.npy'

        save_path = os.path.join(os.path.dirname(self.data_path_csv),
                                filename)

        if len(self.focus_ids) < len(self.sem2id):
            if self.read_cache and os.path.exists(save_path):
                if self.verbose:
                    print('reading {}'.format(save_path)) 
                indices = np.load(save_path).tolist()
            else: # create and save indices 
                indices = [] 
                for index, row in tqdm(self.data_path.iterrows()):
                    path = row['semantic_label']
                    # some checks
                    assert os.path.exists(path), \
                        'semantic_label path {} does not exist'.format(path)
                    assert '.pkl' in path, 'Expected {} to be a pkl file'.\
                            format(path)
                    # checking if desired labels are in current dataset 
                    with open(path, 'rb') as f:
                        labels = pickle.load(f)
                        if any([element in labels.flatten() for element in \
                                list(self.focus_ids)]):
                            # save index
                            indices.append(index)
                # save indices at a given location
                if self.verbose:
                    print('saving to cache file {}'.format(save_path))
                np.save(save_path, np.array(indices))
                if self.verbose:
                    print('saved.')

        else: # all categories are kept, and no need to be saved
            indices = list(range(len(self.data_path)))
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
        batch = []

        if self.random_draw:
            # reset read_index as a random number in [0, length-1]
            self.read_index = random.sample(range(len(self.available_indices)-1),
                                            self.batch_size)
            prod = self.data_path.iloc[[self.available_indices[element] for element in self.read_index]]
        else:
            #increment read_index
            if self.read_index + self.batch_size - 1 <= len(self.available_indices)-1:
                read_ind_seq = range(self.read_index, self.read_index + self.batch_size)
                prod = self.data_path.iloc[[self.available_indices[element] for element in read_ind_seq]]
                self.read_index += self.batch_size
            else:
                raise StopIteration
        
        for idx, element in prod.iterrows():
            rgb = imread(element['rgb'])
            gaze = imread(element['gaze'])
            with open(element['semantic_label'], 'rb') as f:
                sem_label = pickle.load(f)
            
            # zeroing out the classes not in the selected classes 
            target = np.zeros_like(sem_label)
            for cat in list(self.focus_ids):
                vert, horiz = np.where(sem_label == cat)
                target[vert, horiz] = cat
            
            if self.gt_type == 'bbox':
                # TODO: yet to be implemented
                pass
        
            batch.append((rgb, gaze, target))
        
        return batch


if __name__=='__main__':
    saldata = SalconDataloaderInHouse('../data/train_rgb_gaze_sem.csv', '../data/semid.csv')
