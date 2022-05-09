import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils

from torch.autograd import Variable
import numpy as np
from read_data import sceneDisp
import torch.optim as optim

from gc_net import *
from loss import MonodepthLoss
from transforms import image_transforms
from python_pfm import *
from sacred import Experiment
from sacred.observers import FileStorageObserver
from tensorboardX import SummaryWriter
#torch.backends.cudnn.enabled = False


normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}


tsfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize(**normal_mean_var)])
tsfm2=transforms.Compose([transforms.ToTensor()])
data_transform = image_transforms(
        mode='train',
        augment_parameters=[0.8, 1.2, 0.5, 2.0, 0.8, 1.2],
        do_augmentation=True,
        size = (128,384))

h=128
w=384
maxdisp=96 
batch=1
net = Semi_Net(h,w,maxdisp)
net=torch.nn.DataParallel(net).cuda()
optimizer = optim.Adam(net.parameters(), lr=0.0001,betas=(0.9, 0.999))

summary_writer = SummaryWriter(f'./result_nomask/02')
ex = Experiment(name=f'02', interactive=True)
ex.observers.append(FileStorageObserver.create(f'./result_nomask/02'))

def adjust_learning_rate(optimizer, epoch):
    lr = 0.0001
    if epoch>100:
        lr = 0.00001
    #print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#train
def train(epoch_total,loadstate):
    dataset = sceneDisp('/data/kumari/amdim/GC_net/data/training/','train',tsfm)
    #loss_fn= F.smooth_l1_loss()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True,num_workers=1)

    imageL = Variable(torch.FloatTensor(1).cuda())
    imageR = Variable(torch.FloatTensor(1).cuda())
    dispL = Variable(torch.FloatTensor(1).cuda())

    loss_list=[]
    #print(len(dataloader))
    start_epoch=0
    if loadstate==True:
        checkpoint = torch.load('checkpoint_nomask/ckpt47600.t7')
        net.load_state_dict(checkpoint['net'])
        start_epoch = 0
    global_step = 0
    
    for epoch in range(start_epoch,epoch_total):
        #net.train()
        data_iter = iter(dataloader)
        print('\nEpoch: %d' % epoch)
        train_loss=0
        acc_total=0
        for step in range(len(dataloader)-1):
            global_step +=1
            print('----epoch:%d------step:%d------' %(epoch,step))
            data = next(data_iter)
           
            randomH = np.random.randint(30,200)
            randomW = np.random.randint(0, 600)
            print('mask shape before data loader is--------------- ',data['mask'].shape)
            imageL = data['imL'][:,:,randomH:randomH+h,randomW:randomW+w].cuda()
            imageR = data['imR'][:, :,randomH:randomH+h, randomW:randomW+w].cuda()
            dispL = data['dispL'][:, :, randomH:randomH+h, randomW:randomW+w].cuda()
            mask = data['mask'][:, :, randomH:randomH+h, randomW:randomW+w].cuda()
            #normdisp = data['normdisp'][:, :, randomH:randomH+h, randomW:randomW+w].cuda()

            imageL.requires_grad = True
            imageR.requires_grad = True
            dispL.requires_grad = True


            print('left image is ',imageL)    
            print('true disparity is .......................',dispL,type(dispL),dispL.shape)
            #mask = mask.squeeze(0)
            optimizer.zero_grad()

            probl = net(imageL,imageR)
            
            #disp = torch.cat((probl,probr),dim=1)

            print('predicted disparity is', probl,mask)
            print('about max main ',torch.max(imageL),torch.min(imageL),torch.max(imageR),torch.min(imageR),torch.max(dispL),torch.min(dispL),torch.max(probl),torch.min(probl))
            
            #SUPERVISED LOSS
            #loss = F.smooth_l1_loss(probl[mask].float(),dispL[mask].float(),size_average=True)
            loss = torch.nn.L1Loss()(probl[mask].float(),dispL[mask].float())
            print('loss is ######', loss,train_loss,step)
           
            train_loss+=loss.data
        
            loss.backward()
            optimizer.step()
            print('=======loss value for every step=======:%f' % (loss.data))
            print('=======average loss value for every step=======:%f' %(train_loss/(global_step+1)))
            #resultL=prob.view(batch,1,h,w)
            #probl[~mask]=-1
            diffL=torch.abs(probl[mask].data.cpu()-dispL[mask].data.cpu())
            print('absolute difference in disparity is ',diffL)
          

            #print(diffL.shape)
            accuracy3=(torch.sum(diffL<3))/float(h*w*batch)
            accuracy1=((torch.sum(diffL<1))/float(h*w*batch))
            accuracy5=(torch.sum(diffL<5))/float(h*w*batch)
                 
            acc_total+=accuracy1
            print('====accuracy for the result less than 1 pixels===:%f' %accuracy1)
            #print('====average accuracy for the result less than 3 pixels===:%f' % (acc_total/(step+1)))

            # save
            if global_step%200==0:
                #loss_list.append(train_loss/(step+1))
                summary_writer.add_scalar(tag='loss', scalar_value=loss, global_step=global_step)
                summary_writer.add_scalar(tag='accuracy1', scalar_value=accuracy1, global_step=global_step)
                summary_writer.add_scalar(tag='accuracy3', scalar_value=accuracy3, global_step=global_step)
                summary_writer.add_scalar(tag='accuracy5', scalar_value=accuracy5, global_step=global_step)



            if (global_step>1 and global_step%1400==0):
                print('=======>saving model......')
                state={'net':net.state_dict(),'step':global_step,'optimizer_state_dict': optimizer.state_dict(),
                       'loss_list':loss_list,'epoch':epoch,'accur':acc_total}
                torch.save(state,'checkpoint_nomask/ckpt%d.t7'%global_step)
                torch.save(probl,'result_nomask/probl%d.t7'%global_step)
                torch.save(mask,'result_nomask/mask%d.t7'%global_step)
                #torch.save(dispL,'result_nomask/displ%d.t7'%global_step)
                #probl[~mask]=0 


                rdisp = probl.squeeze(0).data.cpu().numpy()
                min_v = np.min(rdisp).item()
                #print('min_v iss ',min_v)
                range_v = np.max(rdisp) - min_v 
                normalised = (rdisp - min_v) / range_v
                
                normalised = (normalised[0,:,:]*255).astype(np.uint8)

                
                x_ = Image.fromarray(cv2.applyColorMap(normalised, cv2.COLORMAP_JET))
                x_.save('result_nomask/train_resultL_%d.png'%global_step)
               
                dispgt = dispL.squeeze(0).data.cpu().numpy()

                max_disp = np.nanmax(dispgt)
                min_disp = np.nanmin(dispgt)
                disp_normalized = (dispgt - min_disp) / (max_disp - min_disp)
                    
                normalised1 = (disp_normalized[0,:,:]*255).astype(np.uint8)
               
                x_ = Image.fromarray(cv2.applyColorMap(normalised1, cv2.COLORMAP_JET))
                x_.save('result_nomask/gt_%d.png'%global_step)
                
                left = (imageL.squeeze(0).permute(1,2,0).data.cpu().numpy()*255).astype(np.uint8)

                cv2.imwrite('result_nomask/train_imagel_%d.png'%global_step, left, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

 

    fp=open('loss.txt','w')
    for i in range(len(loss_list)):
        fp.write(str(loss_list[i][0]))
        fp.write('\n')
    fp.close()


#test
def test(loadstate):

    if loadstate==True:
        checkpoint = torch.load('ckpt73200.t7')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #start_epoch = checkpoint['epoch']
        #accu=checkpoint['accur']
    net.eval()
    imageL = torch.FloatTensor(1).cuda()
    imageR = torch.FloatTensor(1).cuda()
    dispL = torch.FloatTensor(1).cuda()

    dataset = sceneDisp('/data/kumari/amdim/GC_net/data/training/', 'test',tsfm)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    data_iter = iter(dataloader)
    acc_1 = 0
    acc_2 = 0
    acc_3 = 0
    count = 0
    for i in range(len(dataloader)-1):
        data = next(data_iter)
        count = count +1

        randomH = np.random.randint(100, 200)
        randomW = np.random.randint(0, 600)
        #print('test',data_iter)
        imageL = data['imL'][:, :, randomH:(randomH + h), randomW:(randomW + w)].float().cuda()
        #print('size of image left ',imageL.shape)
        imageR = data['imR'][:, :, randomH:(randomH + h), randomW:(randomW + w)].float().cuda()
        dispL = data['dispL'][:, :, randomH:(randomH + h), randomW:(randomW + w)].cuda()
        mask = data['mask'][:, :, randomH:(randomH + h), randomW:(randomW + w)].cuda()


        print('imagel shape  is ',data['imL'].shape)

        with torch.no_grad():
            dispLL=net(imageL,imageR)

        #dispLL[~mask] = 0

        rdisp = dispLL.data.cpu().numpy()

        min_v = torch.min(dispLL).item()
        range_v = torch.max(dispLL).item() - min_v
        normalised = (rdisp - min_v) / range_v
        normalised = (rdisp[0,0,:,:]*255).astype(np.uint8)
        x_ = Image.fromarray(cv2.applyColorMap(normalised, cv2.COLORMAP_JET))
        x_.save('result1/test_resultL_%d.png'%count)


        #dispLL=torch.sum(resultL.mul(loss_mul_test),1)
        
        print('predicted disp is', dispLL)
        print('true groundtruth is ',dispL)
        print('left image is ',imageL)
        #normalised[~mask] = 0
        #torch.save(dispLL,'result1/disp%d.t7'%count)


        diff = torch.abs(dispLL[mask] - dispL[mask])  # end-point-error
        print('difference in disparity tensor ',diff,diff.shape,dispLL.shape)

        accuracy2 = torch.sum(diff<2)/float(128*384)*100 
        accuracy3 = torch.sum(diff<3)/float(128*384)*100
        accuracy1 = torch.sum(diff<1)/float(128*384)*100
        acc_3+=accuracy3
        acc_1+=accuracy1
        acc_2+=accuracy2

        print('test 3,1,2 accuracies:%f %f %f' %(accuracy3,accuracy1,accuracy2))

        rdisp = dispL.squeeze(0).data.cpu().numpy()
        min_v = np.min(rdisp)
        range_v = np.max(rdisp)-min_v
        normalisedd = (rdisp - min_v) / range_v

        #normalisedd= dispLL.data.cpu().numpy()
        normalisedd = (normalisedd[0,:,:]*255).astype(np.uint8)
        x_d = Image.fromarray(cv2.applyColorMap(normalisedd, cv2.COLORMAP_JET))
        x_d.save('result1/gtd_%d.png'%count)

        img = np.transpose((imageL[0,:,:,:].data.cpu().numpy()*255).astype(np.uint8),(1,2,0))
        #print('before saving image ',img.shape)
        cv2.imwrite('result1/test_org_img_%d.png'%count, img)
        #vutils.save_image(tensor=255*img, filename=f'result/test_org_{i}_pred.png')

        #return dispLL
    print('accuracy for 3 pixel is',acc_3,i,acc_3/count)
    print('accuracy for 1 pixel is',acc_1,acc_1/count)
    print('accuracy for 2 pixel is',acc_2,acc_2/count)



def main():
    epoch_total=700
    load_state=False
    train(epoch_total,load_state)
    #test(load_state)


if __name__=='__main__':
    main()
