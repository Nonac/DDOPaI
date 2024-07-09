import abc
import json
import os
import random

from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms



class concepts(abc.ABC):
    def __init__(self,working_dir,label):
        self.PATH = working_dir
        self.label = label
        
    def get_files_in_dir(self, dir):
        selected = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                selected.append(os.path.join(root, file))
        return selected
        
    def get_stitching_patches(self,config):
        selected = {}
        for root, dirs, files in os.walk(self.PATH):
            for dir in dirs:
                waiting_list=self.get_files_in_dir(root + dir+config.stitching_ratio)
                selected[dir]=[]
                waiting_list=self.get_files_in_dir(root + dir+'/stitching_30000_0.')
            break
        imgs, y = self.get_concept_iterator(selected)
        imgs,y=self.get_order_imgs(imgs,y,config.num_concept,config.num_classes,len(self.get_labels()))
        return imgs, y
        
    
    def get_concept_iterator(self, selected):
        labels = self.get_labels()
        imgs = []
        y = []
        for k, v in selected.items():
            for idx, label in enumerate(labels):
                if k ==label:
                    imgs += v
                    for _ in range(len(v)):
                        y.append(idx)
        return imgs, y

    

    
    def get_concept_patches_fixed_class_concept(self, num_concept,num_class):

        dict = self.get_ACE_results()
        selected = {}
        for k, v in dict.items():
            selected[k] = []
            for i in range(num_class):
                selected[k] += self.get_concept_patches(k, v[i]['name'] + "_patches")[:num_concept]

        imgs, y = self.get_concept_iterator(selected)
        imgs,y=self.get_order_imgs(imgs,y,num_concept,num_class,len(self.get_labels()))

        return imgs, y
    
    def get_order_imgs(self,imgs,y,num_concept,num_class,total_class):
        imgs_new=[]
        y_new=[]
        for i in range(num_class*num_concept):
            for j in range(total_class):
                imgs_new.append(imgs[i+j*num_concept*num_class])
                y_new.append(y[i+j*num_concept*num_class])              
        return imgs_new,y_new
            
        

    def get_ACE_results(self):
        dict = {}
        for concept in self.get_labels():
            with open(self.PATH + concept + '/results_summaries/ace_results.txt', "r") as f:
                results = f.readlines()
                dict[concept] = []
                for idx, line in enumerate(results[4:]):
                    dict[concept].append({
                        'name': line.split(':')[0] + '_' + line.split(':')[1],
                        'cav_acc': float(line.split(':')[2].split(',')[0]),
                        '_tcav_score': float(line.split(':')[2].split(',')[1])
                    })
        return dict
    
    def get_labels(self):
        labels = []
        f = open(self.label)
        line = f.readline()
        while line:
            labels.append(line.replace('\n', ''))
            line = f.readline()
        f.close()
        return list(labels)
    
    def get_concept_patches(self, cls_name, patch_name):
        res = []
        for root, dirs, files in os.walk(self.PATH + cls_name + '/concepts/' + patch_name):
            for file in files: res.append(os.path.join(root, file))
        return res


normalize = torchvision.transforms.Normalize(mean=[0.48024578664982126, 0.44807218089384643, 0.3975477478649648],
                                     std=[0.2769864069088257, 0.26906448510256, 0.282081906210584])

preprocess = torchvision.transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ])

def default_loader(path):
    img_pil = Image.open(path)
    img_tensor = preprocess(img_pil)
    return img_tensor


class concepts_set(torch.utils.data.Dataset):
    def __init__(self, file_train, number_train, loader=default_loader):
        self.images = file_train
        self.target = number_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img, target

    def __len__(self):
        return len(self.images)



def get_tiny_imagenet_condept_dataloader(imgs,y, train_batch_size, num_workers=16, root='../data'):
    concept_set = torch.utils.data.DataLoader(
        dataset=concepts_set(imgs, y),
        batch_size=train_batch_size,
        num_workers=num_workers,
        pin_memory=True,
        sampler=None,
        drop_last=False,
        shuffle=True
    )
    return concept_set
    

