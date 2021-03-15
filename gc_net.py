
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *


class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        #self.conv1=nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1)
        #self.elu1=nn.ELU()
        #self.conv2=nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1)
        #self.elu2=nn.ELU()
        #self.conv3=nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1)
        #self.elu3=nn.ELU()
        #self.conv4=nn.Conv2d(96,96,kernel_size=3,stride=1,padding=1)
        #self.elu4=nn.ELU()
        #self.conv5=nn.Conv2d(96,1,kernel_size=3,stride=1,padding=1)
        #self.elu5 = nn.Sigmoid()
        self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1])).cuda()

    #def forward(self, x):
        #out = self.elu1(self.conv1(x))
        #out = self.elu2(self.conv2(out))
        #out= self.elu3(self.conv3(out))
        #out= self.elu4(self.conv4(out))
        #out= self.elu5(self.conv5(out))
        #return out

        #self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1, maxdisp,1,1])).cuda()

    def forward(self, x):
        out = torch.sum(x*self.disp.data,1, keepdim=True)
        return out


class Encoder(nn.Module):  #basic block for Conv2d
    def __init__(self,in_planes,planes,out_planes,stride=1):
        super(Encoder,self).__init__()
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        #self.conv1.weight.data.normal_(0.0, math.sqrt(2. / (3*3*planes)))
        self.elu1 = nn.ELU()
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.conv2.weight.data.normal_(0.0, math.sqrt(2. / (3*3*planes)))
        self.shortcut=nn.Sequential()
        self.bn3 = nn.BatchNorm2d(planes)
        self.elu2 = nn.ELU()
    def forward(self, x):
        out=self.elu1(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out = out+x  #out = out + x
        out=self.elu2(self.bn3(out))
        return out

class ThreeDConv(nn.Module):  #Block for 3d convolution
    def __init__(self,in_planes,mid_planes,out_planes,step):
        super(ThreeDConv, self).__init__()
        
        self.conv1 = nn.Conv3d(in_planes, mid_planes, kernel_size=3, stride=1, padding=1,bias=True)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        #self.conv1.weight.data.normal_(0.0, math.sqrt(2. / (3*3*mid_planes)))
        self.elu1 = nn.ELU()
        
        self.conv2 = nn.Conv3d(mid_planes, mid_planes, kernel_size=3, stride=1, padding=1,bias=True)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.elu2 = nn.ELU()
        #self.conv2.weight.data.normal_(0.0, math.sqrt(2. / (3*3*mid_planes)))

        self.conv3=nn.Conv3d(mid_planes,out_planes,kernel_size=3,stride=2,padding=1,bias=True)
        self.bn3 = nn.BatchNorm3d(out_planes)
        #self.conv3.weight.data.normal_(0.0, math.sqrt(2. / (3*3*out_planes)))
        self.elu3=nn.ELU()

    def forward(self, x):
        out=self.elu1(self.bn1(self.conv1(x)))
        out1=self.elu2(self.bn2(self.conv2(out)))
        out=self.elu3(self.bn3(self.conv3(out1)))
        return out1,out
    
class DeConv(nn.Module):   #Block for 3d transpose convolution
    def __init__(self,in_channel,mid_channel,out_channel,step):
        super(DeConv,self).__init__()
        self.deconv1=nn.ConvTranspose3d(in_channel,out_channel,kernel_size=3,stride=2,padding=1,output_padding=1,bias=True)
        self.elu=nn.ELU()
        self.bn1 = nn.BatchNorm3d(out_channel)
        self.bn2= nn.BatchNorm3d(out_channel)
        self.elu1 = nn.ELU()
        #self.deconv1.weight.data.normal_(0.0, math.sqrt(2. /(3*3*out_channel)))
       
        
    def forward(self,x):
        deconv3d=self.elu(self.bn1(self.deconv1(x[0])))
        #print('transpose conv 3d shape is ',deconv3d.shape)
        deconv3d = self.elu1(self.bn2(deconv3d+x[1]))
        return deconv3d
        
        


class SemiSup_Net(nn.Module):
    def __init__(self,block,block_3d,DeConv,num_block,height,width,maxdisp):
        super(SemiSup_Net, self).__init__()
        self.height=height
        #self.dr = disparityregression(96).cuda()
        self.width=width
        self.maxdisp=int(maxdisp)
        self.in_planes=32
        #first two conv2d
        self.conv0=nn.Conv2d(3,32,kernel_size=5,stride=2,padding=2,bias=True)
        #self.conv0.weight.data.normal_(0.0, math.sqrt(2. / (5*5*32)))
        self.bn0 = nn.BatchNorm2d(32)
        self.elu0=nn.ELU()
        #res block
        self.res_block1=self._make_layer(block,self.in_planes,32,32,1)  # 7 times
        self.res_block2=self._make_layer(block,self.in_planes,32,32,1)
        self.res_block3=self._make_layer(block,self.in_planes,32,32,1)
        self.res_block4=self._make_layer(block,self.in_planes,32,32,1)
        self.res_block5=self._make_layer(block,self.in_planes,32,32,1)
        self.res_block6=self._make_layer(block,self.in_planes,32,32,1)
        self.res_block7=self._make_layer(block,self.in_planes,32,32,1)
        self.res_block8=self._make_layer(block,self.in_planes,32,32,1)
       
        #last conv2d
        self.conv1=nn.Conv2d(32,32,kernel_size=3,stride=1,padding=1,bias=True)
        #self.conv1.weight.data.normal_(0.0, math.sqrt(2. / (3*3*32)))
        self.bn1 = nn.BatchNorm2d(32)
        

        #conv3d
        self.block_3d_5=nn.Conv3d(128,128,kernel_size=3,stride=1,padding=1,bias=True)
        #self.block_3d_5.weight.data.normal_(0.0, math.sqrt(2. / (3*3*128)))
        self.bn3d_5 = nn.BatchNorm3d(128)
        self.elu1 = nn.ELU()
        
        self.block_3d_6=nn.Conv3d(128,128,kernel_size=3,stride=1,padding=1,bias=True)
        self.bn3d_6 = nn.BatchNorm3d(128)
        #self.block_3d_6.weight.data.normal_(0.0, math.sqrt(2. / (3*3*128)))
        self.elu2=nn.ELU()
        
        self.upsampler = nn.ConvTranspose3d(32,1,kernel_size=3,stride=2,padding=1,output_padding=1,bias=True)
        #self.upsampler.weight.data.normal_(0.0, math.sqrt(2. / (3*3*1)))
        #self.bnlast =  nn.BatchNorm3d(1)
        #self.tanh = nn.ELU()

        



        #conv3d sub_sample block
        self.block_3d_1 = self._make_layer(block_3d,64,32,64,1)
        self.block_3d_2 = self._make_layer(block_3d,64, 64, 64,2)
        self.block_3d_3 = self._make_layer(block_3d,64, 64, 64,3)
        self.block_3d_4 = self._make_layer(block_3d,64, 64, 128,4)
        
        #Deconv3d sub_sample block
        self.block_t3d_1 = self._make_layer(DeConv,128,64,64,1)
        self.block_t3d_2 = self._make_layer(DeConv,64, 64, 64,2)
        self.block_t3d_3 = self._make_layer(DeConv,64, 64, 64,3)
        self.block_t3d_4 = self._make_layer(DeConv,64, 32, 32,4)
        
     

    def forward(self, imgLeft,imgRight):
        imgl0=self.elu0(self.bn0(self.conv0(imgLeft)))
        imgr0=self.elu0(self.bn1(self.conv0(imgRight)))

        imgl_block=self.res_block1(imgl0)
        imgr_block=self.res_block1(imgr0)

        imgl_block=self.res_block2(imgl_block)
        imgr_block=self.res_block2(imgr_block)


        imgl_block=self.res_block3(imgl_block)
        imgr_block=self.res_block3(imgr_block)

        imgl_block=self.res_block4(imgl_block)
        imgr_block=self.res_block4(imgr_block)

        imgl_block=self.res_block5(imgl_block)
        imgr_block=self.res_block5(imgr_block)

        imgl_block=self.res_block6(imgl_block)
        imgr_block=self.res_block6(imgr_block)

        imgl_block=self.res_block7(imgl_block)
        imgr_block=self.res_block7(imgr_block)

        imgl_block=self.res_block8(imgl_block)
        imgr_block=self.res_block8(imgr_block)
        
        imgl1=self.bn1(self.conv1(imgl_block))
        imgr1=self.bn1(self.conv1(imgr_block))
        

        #cost_volum = self.cost_volume(imgl1,imgr1,imgl1.size(2),imgl1.size(3))
        #costr = Variable(torch.FloatTensor(imgr1.size()[0], imgr1.size()[1]*2, int(self.maxdisp/2),  int(imgr1.size()[2]),  imgr1.size()[3]).zero_(), volatile= not self.training).cuda()

        #for i in range(int(self.maxdisp/2)):
        #    if i > 0 :
        #     costr[:, :imgr1.size()[1], i, :,i:]   = imgr1[:,:,:,i:]
        #     costr[:, imgr1.size()[1]:, i, :,i:] = imgl1[:,:,:,:-i]
        #    else:
        #     costr[:, :imgr1.size()[1], i, :,:]   = imgr1
        #     costr[:, imgr1.size()[1]:, i, :,:]   = imgl1
        #costr = costr.contiguous()
        #print('cost volume shape is ',costr.shape)

        #cost_volum = self.cost_volume(imgl1,imgr1,imgl1.size(2),imgl1.size(3))
        cost = Variable(torch.FloatTensor(imgl1.size()[0], imgl1.size()[1]*2, int(self.maxdisp/2),  int(imgl1.size()[2]),  imgl1.size()[3]).zero_(), volatile= not self.training).cuda()

        for i in range(int(self.maxdisp/2)):
            if i > 0 :
             cost[:, :imgl1.size()[1], i, :,i:]   = imgl1[:,:,:,i:]
             cost[:, imgl1.size()[1]:, i, :,i:] = imgr1[:,:,:,:-i]
            else:
             cost[:, :imgl1.size()[1], i, :,:]   = imgl1
             cost[:, imgl1.size()[1]:, i, :,:]   = imgr1
        cost = cost.contiguous()
        print('cost volume shape is ',cost.shape)

        
        

        #conv3d block
        _12b,conv3d_block_1=self.block_3d_1(cost)
        #_12rb,conv3d_block_r1=self.block_3d_1(costr)
      
        _13b,conv3d_block_2=self.block_3d_2(conv3d_block_1)
        #_13rb,conv3d_block_r2=self.block_3d_2(conv3d_block_r1)
       
        _14b,conv3d_block_3=self.block_3d_3(conv3d_block_2)
        #_14rb,conv3d_block_r3=self.block_3d_3(conv3d_block_r2)
        
        _15b,conv3d_block_4=self.block_3d_4(conv3d_block_3)
        #_15rb,conv3d_block_r4=self.block_3d_4(conv3d_block_r3)

        
        conv3d_block_5=self.elu1(self.bn3d_5(self.block_3d_5(conv3d_block_4)))
        conv3d_block_6=self.elu2(self.bn3d_6(self.block_3d_6(conv3d_block_5)))
        #print('shapes off intermediate layers',_15b.shape,_14b.shape,_13b.shape,_12b.shape)

        #conv3d_block_r5=self.elu1(self.bn3d_5(self.block_3d_5(conv3d_block_r4)))
        #conv3d_block_r6=self.elu2(self.bn3d_6(self.block_3d_6(conv3d_block_r5)))

        
       
        #deconv
        deconv3d=self.block_t3d_1([conv3d_block_6,_15b])
        #deconv3dr=self.block_t3d_1([conv3d_block_r6,_15rb])

        
        deconv3d=self.block_t3d_2([deconv3d,_14b])
        #deconv3dr=self.block_t3d_2([deconv3dr,_14rb])

              
        
        deconv3d=self.block_t3d_3([deconv3d,_13b])
        deconv3d=self.block_t3d_4([deconv3d,_12b])

        #deconv3dr=self.block_t3d_3([deconv3dr,_13rb])
        #deconv3dr=self.block_t3d_4([deconv3dr,_12rb])


        #last deconv3d
        last_deconv3d=self.upsampler(deconv3d)

        #last_deconv3d = F.upsample(deconv3d, scale_factor=2, mode='trilinear')
        #last_deconv3dr=self.upsampler(deconv3dr)

        print('last conv 3d shape is ',last_deconv3d.shape)
        
        
        
        
        

        #print('last',last_deconv3d.shape)
        
        outl=last_deconv3d.view(1, self.maxdisp, self.height, self.width)
        #outr=last_deconv3dr.view(1, self.maxdisp, self.height, self.width)

        print('shape of last layer out put is ',outl.shape)
        probl=F.softmax(outl,1)
        predl = disparityregression(96)(probl)
        
        #probr=F.softmax(outr,1)
        #predr = disparityregression(96)(probr)

        return predl



        



    def _make_layer(self,block,in_planes,mid_planes,out_planes,step):
        #strides=5
        layers=[]
        
        layers.append(block(in_planes,mid_planes,out_planes,step))
        return nn.Sequential(*layers)


    def cost_volume(self,imgl,imgr,xh,xw):
        xx_list = []
        pad_opr1 = nn.ZeroPad2d((0, self.maxdisp, 0, 0))
        xleft = pad_opr1(imgl)
        for d in range(self.maxdisp):  # maxdisp+1 ?
            pad_opr2 = nn.ZeroPad2d((d, self.maxdisp - d, 0, 0))
            xright = pad_opr2(imgr)
            xx_temp = torch.cat((xleft, xright), 1)
            xx_list.append(xx_temp)
        xx = Variable(torch.cat(xx_list, 1))
        xx = xx.view(1, self.maxdisp, 64, xh, xw+self.maxdisp)
        xx0=xx.permute(0,2,1,3,4)
        xx0 = xx0[:, :, :, :, :xw]
        return xx0

def loss(xx,loss_mul,gt):
    loss=torch.sum(torch.sqrt(torch.pow(torch.sum(xx.mul(loss_mul),1)-gt,2)+0.00000001)/256/(256+128))
    return loss

def Semi_Net(height,width,maxdisp):
    return SemiSup_Net(Encoder,ThreeDConv,DeConv,[8,1],height,width,maxdisp)





