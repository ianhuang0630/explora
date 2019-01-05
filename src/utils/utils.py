"""
@author: Ian Huang
Utilities for processing the data 
"""
import torch
import numpy as np
from skimage import io, transform

class Rescale(object):
    def __init__(self, output_size, target_type='pix'):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        assert target_type=='pix' or target_type=='bbox'
        self.target_type = target_type

    def __call__(self, sample):
       	image, gaze, target = sample['rgb'], sample['gaze'], sample['target']
         
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        gz = transform.resize(gaze, (new_h, new_w))
        if self.target_type == 'pix':
            tg = transform.resize(target)
        else:
            raise ValueError('Havent yet implemented bounding box ground truth')
            # TODO: implement this
            # landmarks = landmarks * [new_w / w, new_h / h]
            # h and w are swapped for landmarks because for images,
            # x and y axes are axis 1 and 0 respectively

        return {'rgb': img, 'gaze': gz, 'target': tg} 

class RandomCrop(object):
    def __init__(self, output_size, target_type='pix'):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size 
        
        assert target_type=='pix' or target_type=='bbox'
        self.target_type = target_type

    def __call__(self, sample):
	image, gaze, target = sample['rgb'], sample['gaze'], sample['target']
        
	h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img = image[top: top + new_h,
                left: left + new_w, :]
        gz = gaze [top:top+new_h, left:left+new_w]
        if self.target_type == 'pix':
            tg = target[top: top+new_h, left:left+new_w]
        else:
            raise ValueError('Havent yet implemented bounding box ground truth')
            # TODO: implement this
        return {'rgb': img, 'gaze': gz, 'target': tg}

class ToTensor(object):
    def __init__(self, target_type='pix'):
        assert target_type=='pix' or target_type=='bbox'
        self.target_type = target_type

    def __call__(self, sample):
        image, gaze, target = sample['rgb'], sample['gaze'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        img = torch.from_numpy(image)
        gz = torch.from_numpy(gaze)
        if self.target_type=='pix':
            tg = torch.from_numpy(target)
        else:
            raise ValueError('Havent yet implemented bounding box ground truth')
            # TODO: implement this
        return {'image': img, 'gaze': gz, 'target': tg} 

