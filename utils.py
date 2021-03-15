'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from PIL import Image

#class disparityregression(nn.Module):
#    def __init__(self, maxdisp):
#        super(disparityregression, self).__init__()
        #self.conv1=nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1)
        #self.elu1=nn.ELU()
        #self.conv2=nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1)
        #self.elu2=nn.ELU()
        #self.conv3=nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1)
        #self.elu3=nn.ELU()
        #self.conv4=nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1)
        ##self.elu4=nn.ELU()
        #self.conv5=nn.Conv2d(96,1,kernel_size=3,stride=1,padding=1)
        #self.elu5 = nn.Sigmoid()
        #self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1])).cuda()

    #def forward(self, x):
     #   out = self.elu1(self.conv1(x))
     #   out = self.elu2(self.conv2(out))
     #   out= self.elu3(self.conv3(out))
     #   out= self.elu4(self.conv4(out))
     #   out= self.elu5(self.conv5(out))

        #out = torch.sum(x*self.disp.data,1, keepdim=True)

      #  return out

#class disparityregression(nn.Module):
#    def __init__(self, maxdisp):
#        super(disparityregression, self).__init__()
#        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1])).cuda()

#    def forward(self, x):
#        out = torch.sum(x*self.disp.data,1, keepdim=True)
#        return out


def depth_read(filename):
	
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    print(np.unique(depth_png))
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    #depth[depth_png == 0] = -1.
    return depth




