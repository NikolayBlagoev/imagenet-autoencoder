import torch.nn as nn
import torch.nn.parallel as parallel

from . import vgg, resnet

def BuildAutoEncoder(args):

    

    
    model = resnet.ResNetAutoEncoder([3, 4, 6, 3], True)
    

    return model.to("cuda")