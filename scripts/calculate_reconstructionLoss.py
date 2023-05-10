import torch
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms

@torch.no_grad()
def cal_rec_loss(t1,t2):
    loss=F.l1_loss(t1,t2)
    return loss

@torch.no_grad()
def cal_total_image_rec_loss(dir1,dir2):
    assert os.path.exists(dir1)
    assert os.path.exists(dir2)
    list1=os.listdir(dir1)
    list2=os.listdir(dir2)
    assert len(list1)==len(list2)
    amount=len(list1)
    total_loss=0
    for i in range(amount):
        name1=list1[i]
        name2=list2[i]
        path1=os.path.join(dir1,name1)
        path2=os.path.join(dir2,name2)
        image1 = Image.open(path1)
        image2 = Image.open(path2)
        toTensor=transforms.ToTensor()
        t1=toTensor(image1)
        t2=toTensor(image2)
        loss=cal_rec_loss(t1,t2)
        total_loss+=loss.item()
        image1.close()
        image2.close()
    print(total_loss)
    print(total_loss/amount)
if __name__ == '__main__':
    dir1=r'C:\Users\赵鑫\Desktop\final\buchong_shiyan\完整数据\vg_ground_truth_256'
    dir2=r'C:\Users\赵鑫\Desktop\final\buchong_shiyan\完整数据\vg_rec_vqgan_mytrain_in_vg'

    cal_total_image_rec_loss(dir1,dir2)