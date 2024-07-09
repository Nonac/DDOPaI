import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from models.base.vit import ViT
from ACE.pytorch.ace_processing import ACE
from utils.dict import DictToObj
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader


config_dict={
    "size":32,
    "patch":4,
    "dimhead":512,
    "FEATHER_NAMES":["transformer"],
    "LABEL_PATH":"cifar10_labels.txt",
    "WORKING_DIR":"/mnt/ssd_3/DDOPaI/tmp/cifar10_vit",
    "MAX_IMGS":30,
    "MIN_IMGS":10,
    "batch_size":40,
    "seed":1,
    "RANDOM_CONCEPT":"random_discovery",
    "RANDOM_SOURCE":"tmp/random",
    "MEAN":[0.4914, 0.4822, 0.4465],
    "STD": [0.2023, 0.1994, 0.2010],
    "N_SEGMENTS":[5,10,15],
    "IMAGE_SHAPE":[32,32],
    "NUM_RANDOM_EXP":10
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True



if __name__=="__main__":
    config=DictToObj(config_dict)
    setup_seed(config.seed)
    net = ViT(
    image_size = config.size,
    patch_size = config.patch,
    num_classes = 10,
    dim = int(config.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
    net.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    
    
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transforms.Compose([transforms.Resize(32),transforms.ToTensor()]))
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=8)
    


    cd=ACE(config,net,samples=[],target=[],labels=config.LABEL_PATH,dataloader=trainloader)
    cd.concept_discovery()
    print("finished concept discovery")



    
    
        
