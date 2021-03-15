from torch.utils.data import Dataset, DataLoader
import pickle
import cv2
from python_pfm import readPFM
import numpy as np
import torch
#from utils import read_depth
import os
from PIL import Image
from utils import depth_read

class sceneDisp(Dataset):

    def __init__(self, root_dir,settype,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.settype = settype
        self.transform = transform

        self.kitti_frame = []
        if self.settype == 'train':
            self.filename = 'kitti_GC_train.txt'
        else:
            self.filename = 'kitti_GC_test.txt'
        with open(self.filename, "r") as file:
            lines = file.readlines()
            self.kitti_frame=lines
        #print(self.kitti_frame)

        self.root_dir = root_dir
        self.transform1 = transform


    def __len__(self):
        return len(self.kitti_frame)

    def __getitem__(self, idx):
        #print('anchor is ',os.path.join(self.root_dir+'/image_2/',self.kitti_frame[idx]))
        imageL = Image.open(os.path.join(self.root_dir+'/image_2/',self.kitti_frame[idx].rstrip('\n')))
        print('left image in read_data file ',np.array(imageL).shape)
        #imageL = imageL[:370,:1200]
        #print('left image shape is',imageL.shape)
        imageR = Image.open(os.path.join(self.root_dir+'/image_3/',self.kitti_frame[idx].rstrip('\n')))
        #imageR = imageR[:370,:1200]
        #print(os.path.join(self.root_dir+'/disp_noc_0/',self.kitti_frame[idx]))
        dispL =  np.array(depth_read(os.path.join(self.root_dir+'/disp_noc_0/',self.kitti_frame[idx]).rstrip('\n'))).astype(np.uint8)
        #print('type of disp is ',type(dispL),torch.max(dispL),torch.min(dispL))
        #dispL = dispL[:370,:1200]

        #mask = dispL != 0
        #kernel = np.ones((5,5), np.uint8)
        #erosion = cv2.erode(image, kernel, iterations=1)
        #depth = torch.tensor(cv2.dilate(dispL.numpy(), kernel, iterations=2))
        mask = dispL != 0

        mask = torch.tensor(mask).unsqueeze(0)
        #mask.detach_()
        #depth = depth.unsqueeze(0)
        #print('non scaled disparity is ',depth)
        
        #min_v = torch.min(dispL)
        #range_v = torch.max(dispL) - min_v
        #if range_v > 0:
        #    normalised = (dispL - min_v) / range_v
        #else:
        #    normalised = torch.zeros(dispL.size())
        
        #dispL = normalised
        #normalised[~mask]=0
        #print('scaled disparity is ',normalised)
        #dispL = dispL.reshape(1242,375,1).transpose((2, 0, 1))
        print('mask shape is ',mask.shape)
        
        
        sample = {'imL':imageL, 'imR': imageR, 'dispL': torch.tensor(dispL).float().unsqueeze(0),'mask':mask}
        #print('inside dataloader ',sample['imL'].shape,sample['imR'].shape,sample['dispL'].shape)
        if self.transform1 is not None:
            #sample=self.transform1(sample)
            sample['imL']=self.transform1(sample['imL'])
            sample['imR']=self.transform1(sample['imR'])
        else:
            sample['imL']=torch.tensor(np.array(sample['imL'])).permute(2,0,1)
            sample['imR']=torch.tensor(np.array(sample['imR'])).permute(2,0,1)

        return sample

