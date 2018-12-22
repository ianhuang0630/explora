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
import pandas as pd

def process_gaze_data():
     
    pass

def process_sem_annots(coco, imgids, str_imgids, file_prefix, save_dir_path):
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
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    for i, element in enumerate(tqdm(imgids)):
        this_imgid = coco.getImgIds(imgIds=element)
        img = coco.loadImgs(this_imgid)[0]
        dims = (img['height'], img['width'])
        # loading annotations
        annots_id = coco.getAnnIds(imgIds = this_imgid)
        anns = coco.loadAnns(annots_id)
        # do the following for every category
        masks = [(coco.annToMask(element), element['category_id']) 
                for element in anns]
        label_masks = [element[0]* element[1] for element in masks]
        # overlay on 0's
        canvas = np.zeros(dims)
        for i in range(len(label_masks)):
            # check for duplicates
            this_mask = label_masks[i].copy()
            this_mask[np.where(canvas != 0)] = 0
            canvas = canvas + this_mask
        # saving 
        file_name = file_prefix + str_imgids[i] + '.pkl'
        pickle_path = os.path.join(save_dir_path, file_name)
        pickle.dump(canvas , open(pickle_path, 'wb'))    
        sem_labels.append(pickle_path)
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
    train_aggregate_paths_loc = '../data/train_rgb_gaze_sem.csv'
    val_aggregate_paths_loc = '../data/val_rgb_gaze_sem.csv'

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
        imgpath = os.path.join(img_path, train_prefix + element + '.jpg')
        assert os.path.exists(imgpath), '{} does not exist'.format(imgpath)
        train_img_paths.append(imgpath)
    
    print('Processing paths for training images')
    val_img_paths = []
    for element in tqdm(val_img_ids):
        imgpath = os.path.join(img_path, val_prefix + element + '.jpg')
        assert os.path.exists(imgpath), '{} does not exist'.format(imgpath)
        val_img_paths.append(imgpath)
    
    ## 2) get the gaze data
    train_gaze_paths = []
    for element in tqdm(train_img_ids):
        imgpath = os.path.join(saliency_train_path, train_prefix + element + '.png')
        assert os.path.exists(imgpath), '{} does not exist'.format(imgpath)
        train_gaze_paths.append(imgpath)

    val_gaze_paths = []
    for element in tqdm(val_img_ids):
        imgpath = os.path.join(saliency_val_path, val_prefix + element + '.png')
        assert os.path.exists(imgpath), '{} does not exist'.format(imgpath)
        val_gaze_paths.append(imgpath)

    ## 3) get the segmentation data
    int_id_train = [int(element) for element in train_img_ids]
    int_id_val = [int(element) for element in val_img_ids]
    train_sem_paths  = process_sem_annots(coco_train, int_id_train, 
            train_img_ids, train_prefix, semantic_train_path) 
    val_sem_paths  = process_sem_annots(coco_val, int_id_val, 
            val_img_ids, val_prefix, semantic_val_path) 
   
    # make list of path, and save into training and validation
    # TODO: save: 1)train_img_paths 2) train_gaze_paths, 3) train_sem_paths
    assert len(train_img_paths) == len(train_sem_paths) and \
        len(train_sem_paths) == len(train_gaze_paths), \
        'Lists need to be the same length'
    print('saving paths to {}'.format(train_aggregate_paths_loc))
    data_train = {'rgb': train_img_paths, 
                  'gaze': train_gaze_paths,
                  'semantic_label': train_sem_paths}
    df_train = pd.DataFrame(data=data_train)
    df_train.to_csv(train_aggregate_paths_loc, index=False, 
                    columns=['rgb', 'gaze', 'semantic_label'])
    print('Done.')

    # TODO: save: 1) val_img_paths 2) val_gaze_paths, 3) val_sem_paths
    assert len(val_img_paths) == len(val_sem_paths) and \
        len(val_sem_paths) == len(val_gaze_paths), \
        'Lists need to be the same length'
    print('saving paths to {}'.format(val_aggregate_paths_loc))
    data_val = {'rgb': val_img_paths, 
                  'gaze': val_gaze_paths,
                  'semantic_label': val_sem_paths}
    df_val = pd.DataFrame(data=data_val)
    df_val.to_csv(val_aggregate_paths_loc, index=False, 
                    columns=['rgb', 'gaze', 'semantic_label'])
    print('Done.')

