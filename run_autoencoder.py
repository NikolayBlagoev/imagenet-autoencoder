import os
import sys
import math
import random
import numpy as np
import models.builer as builder

import torch.nn as nn
import torch
from matplotlib import pyplot as plt
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import transforms
import torch
import torchvision
from torchvision.datasets.utils import download_url
from torchvision.transforms import transforms
from tqdm import tqdm
import torchvision
from torchvision.datasets.utils import download_url

val_trans = transforms.Compose([
                    transforms.Resize(256),                   
                    transforms.CenterCrop(224),
                    transforms.ToTensor()
                  ])
# torchvision.set_video_backend("video_reader")
stream = "video"





class ImageDataset(data.Dataset):
    def __init__(self, ann_file, transform=None):
        self.ann_file = ann_file
        self.transform = transform
        self.mask = torch.cat((torch.zeros((3,10,224)), torch.ones((3,186,224)), torch.zeros((3,28,224))), 1) 
        self.init()

    def init(self):
        
        self.im_names = []
        self.targets = []
        with open(self.ann_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(' ')
                self.im_names.append(data[0])
                self.targets.append(int(data[1]))

    def __getitem__(self, index):
        im_name = self.im_names[index]
        target = self.targets[index]

        img = Image.open(im_name).convert('RGB') 
        if img is None:
            print(im_name)
        
        img = self.transform(img)

        return img*self.mask, img*self.mask

    def __len__(self):
        return len(self.im_names)


def load_dict(resume_path, model):
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        model_dict = model.state_dict()
        model_dict.update(checkpoint['state_dict'])
        model.load_state_dict(model_dict)
        del checkpoint
    else:
        sys.exit("=> No checkpoint found at '{}'".format(resume_path))
    return model
val_trans = transforms.Compose([
                    transforms.Resize(224),                   
                    # transforms.CenterCrop(224),
                    transforms.ToTensor()
                  ])

val_dataset = ImageDataset("list/highlights_list.txt", transform=val_trans)
print("MAKING AUTOENCODE")
model = builder.BuildAutoEncoder("")   
print("LOADING STATE DICTIONARY")
model  = load_dict("outputs/010.pth", model)




val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    shuffle=False,
                    batch_size=8,
                   
                    pin_memory=True)

model.eval()
losses = np.array([])
criterion = nn.MSELoss(reduction = 'mean', reduce = False)
with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader)):
           
            
            input = input.cuda(non_blocking=False)
            target = target.cuda(non_blocking=False)

            output = model(input)
            
            loss = criterion(output, target)
            loss = torch.mean(loss, dim = (1,2,3))
            loss = loss.detach().cpu().numpy()
            
            # record loss
            losses = np.concatenate((losses,loss))
np.savetxt('arr.csv', losses, delimiter=',')
plt.plot(losses)
plt.show()


    
    