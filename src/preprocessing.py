"""
@author: Ian
This is a preprocessing script that takes in the raw data and organizes
the data ito a tuple: (image, eyetracking, labels)
"""
import os
import numpy as np
import scipy as sp
from multiprocessing import Pool
from scipy.misc import imread 
from cocoapi.PythonAPI.pycocotools.coco import COCO
from tqdm import tqdm
import pickle

def process_gaze_data():
     
    pass

def process_sem_annots(coco, imgids):
    """ Get semantic labels from COCO datastructure
    Inputs:
        coco: COCO datastructure
        imgids: list of integers, image ids
    Returns:
        sem_labels: list of matrices that hold the semantic labels at each
            pixel of every image.
    """
    sem_labels = []
    print('preparing annotations for images...')
    for element in tqdm(imgids):
        this_imgid = coco.getImgIds(imgIds=element)
        img = coco.loadImgs(this_imgid)[0]
        dims = (img['height'], img['height'])
        # loading annotations
        annots_id = coco.getAnnIds(imgIds = this_imgid)
        anns = coco.loadAnns(annots_id)
        # do the following for every category
        masks = [(coco.annToMask[element], element['category_id']) 
                for element in anns]
        label_masks = [element[0]* element[1] for element in masks]
        # overlay on 0's
        canvas = np.zeros(dims)
        for i in range(len(label_masks)):
            # check for duplicates
            this_mask = label_masks[i].copy()
            this_mask[np.where(canvas != 0)] = 0
            canvas = canvas + this_mask
        sem_labels.append((img['id'], canvas))
    return sem_labels

if __name__ == '__main__':
    
    img_path = '../data/salcon/images'
    saliency_train_path = '../data/salcon/train'
    saliency_test_path = '../data/salcon/test'
    saliency_val_path = '../data/salcon/val'
    
    annots_path = '../data/salcon/coco/labels/annotations'
    annot_name_train = 'instances_train2014.json'
    annot_name_val = 'instances_val2014.json'
    annot_train_path = os.path.join(annots_path, annot_name_train)
    annot_val_path = os.path.join(annots_path, annot_name_val)
    
    # save locations for semantic labeling
    semantic_train_path = '../data/sem_train'
    semantic_val_path = '../data/sem_val'

    # process labels
    # TODO: scan for certain classes: dogs and cats?
    coco_train = COCO(annot_train_path)
    coco_val = COCO(annot_val_path)
    
    # cycle through every single imag eid in train/val of salcon, find
    # corresponding instance annotations
    train_img_ids = os.listdir(saliency_train_path) 
    val_img_ids = os.listdir(saliency_val_path)
    train_prefix = 'COCO_train2014_'
    val_prefix  = 'COCO_val2014_' 
    train_img_ids = [element[len(train_prefix):-(len('.mat'))] 
                    for element in train_img_ids if '.mat' in element]
    val_img_ids = [element[len(val_prefix):-(len('.mat'))] 
                    for element in val_img_ids if '.mat' in element]

    # TODO: parallelize      
    ## 1) get the rgb image
    # getting training images
    train_img_paths = []
    print('Processing paths for training images')
    for element in tqdm(train_img_ids):
        imgpath = os.path.join(img_path, train_prefix + train_img_ids + '.jpg')
        assert os.path.exists(imgpath), '{} does not exist'.format(imgpath)
        train_img_paths.append(imgpath)
    
    print('Processing paths for training images')
    val_img_paths = []
    for element in tqdm(val_img_ids):
        imgpath = os.path.join(img_path, val_prefix + val_img_ids + '.jpg')
        assert os.path.exists(imgpath). '{} does not exist'.format(imgpath)
        val_img_paths.append(imgpath)
    
    ## 2) get the gaze data
     

    ## 3) get the segmentation data
    int_id_train = [int(element) for element in train_img_ids]
    int_id_val = [int(element) for element in val_img_ids]
    sem_train = process_sem_annots(coco_train, int_id_train) 
    sem_val = process_sem_annots(coco_val, int_id_val) 
    
    # now save to file
    train_sem_loc = []
    if not os.path.exist(semantic_train_path):
        os.makedirs(semantic_train_path)
    print('Saving the semantic labelings in .pkl under semantic_train folder...')
    for i in tqdm(range(len(sem_train))):
        # dump the data
        sem_label_mat = sem_train[i][1]
        file_name = train_prefix + train_img_ids[i] + '.pkl'
        pickle_path = os.path.join(semantic_train_path, file_name)
        pickle.dump(sem_label_mat, open(pickle_path, 'wb'))    
        train_sem_loc.append(pickle_path)
         
    val_sem_loc = []
    if not os.path.exist(semantic_val_path):
        os.makedirs(semantic_val_path)
    print('Saving the semantic labelings in .pkl under semantic_val folder...')
    for i in tqdm(range(len(sem_val))):
        # dump the data
        sem_label_mat = sem_val[i][1]
        file_name = val_prefix + val_img_ids[i] + '.pkl'
        pickle_path = os.path.join(semantic_val_path, file_name)
        pickle.dump(sem_label_mat, open(pickle_path, 'wb'))    
        val_sem_loc.append(pickle_path)
    
    
    # make list of path, and save into training and validation
    
    
    pass
