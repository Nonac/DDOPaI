import abc
import json
import os
import random

from PIL import Image
import torch
import torchvision



class concepts(abc.ABC):
    def __init__(self,working_dir,label):
        self.PATH = working_dir
        self.label = label


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
                selected[dir]=[]
                for _ in range(3):
                    waiting_list=self.get_files_in_dir(os.path.join(root,dir,'stitching_30000_'+str(config.stitching_ratio)))
                    waiting_list=self.get_files_in_dir(root + dir+'/stitching_30000_0.')
            break
        imgs, y = self.get_concept_iterator(selected)
        return imgs, y


    def get_concept_patches_fixed_per_class(self, num):

        dict = self.get_ACE_results()
        selected = {}
        for k, v in dict.items():
            selected[k] = []
            for i in range(num):
                selected[k] += self.get_concept_patches(k, v[i]['name'] + "_patches")

        imgs, y = self.get_concept_iterator(selected)

        return imgs, y



    def get_concept_patches_fixed_class_concept(self, num_concept,num_class):

        dict = self.get_ACE_results()
        selected = {}
        for k, v in dict.items():
            selected[k] = []
            for i in range(num_class):
                selected[k] += self.get_concept_patches(k, v[i]['name'] + "_patches")[:num_concept]

        imgs, y = self.get_concept_iterator(selected)

        return imgs, y
    
    def _get_concept_patches_fixed_class_concept(self, num_concept,num_class):

        dict = self.get_ACE_results()
        selected = {}
        for k, v in dict.items():
            selected[k] = []
            for i in range(num_class):
                tmp=i
                while tmp>=len(v):
                    tmp-=len(v)       
                selected[k] += self.get_concept_patches(k, v[tmp]['name'] + "_patches")[:num_concept]
        imgs, y = self.get_concept_iterator(selected)

        return imgs, y
    
    

    def get_concept_patches_fixed_class_concept_rev(self, num_concept,num_class):

        dict = self.get_ACE_results()
        selected = {}
        for k, v in dict.items():
            selected[k] = []
            for i in range(num_class):
                selected[k] += self.get_concept_patches(k, v[-(i+1)]['name'] + "_patches")[:num_concept]

        imgs, y = self.get_concept_iterator(selected)

        return imgs, y

    def get_segment(self,num):
        print("Getting segment........")
        selected = {}
        for root, dirs, files in os.walk(self.PATH):
            for dir in dirs:
                segments=[]
                for i in range(40):
                    for r,d,f in os.walk(root+dir+"/segment/"+str(i+1)):
                        for segment in f:
                            segments.append(r+'/'+ segment)
                selected[dir]=random.sample(segments,num)
            break
        imgs, y = self.get_concept_iterator(selected)

        return imgs, y

    def get_concept_patched_larger_than_cav_acc(self, cav_acc):
        dict = self.get_ACE_results()
        selected = {}
        for k, v in dict.items():
            selected[k] = []
            for i in range(len(v)):
                if v[i]['cav_acc'] >= cav_acc:
                    selected[k] += self.get_concept_patches(k, v[i]['name'] + "_patches")
                else:
                    break

        imgs, y = self.get_concept_iterator(selected)
        return imgs, y

    def get_concept_patches(self, cls_name, patch_name,item_name='concepts'):
        res = []
        for root, dirs, files in os.walk(os.path.join(self.PATH, cls_name , item_name,patch_name)):
            for file in files: 
                if file.endswith(".png"):
                    res.append(os.path.join(root, file))
        return res

    def get_ACE_results(self):
        dict = {}
        for concept in self.get_labels():
            with open(os.path.join(self.PATH, concept, 'results_summaries','ace_results.txt'), "r") as f:
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

    def get_label_from_idx(self, idx):
        labels = self.get_labels()
        if isinstance(idx, int):
            return labels[idx]
        elif isinstance(idx, list):
            return [labels[x] for x in idx]

    def _get_labels_from_json(self):
        with open(self.label_json) as f:
            return json.load(f)




normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

preprocess = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ])


# cifar10_preprocess = torchvision.transforms.Compose([
#             torchvision.transforms.RandomCrop(32, padding=4),
#             torchvision.transforms.RandomHorizontalFlip(),
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

cifar10_preprocess = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.Resize(32),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

def default_loader(path):
    img_pil = Image.open(path)
    img_tensor = preprocess(img_pil)
    return img_tensor

def cifar10_default_loader(path):
    img_pil = Image.open(path)
    img_tensor = cifar10_preprocess(img_pil)
    return img_tensor


class concepts_set(torch.utils.data.Dataset):
    def __init__(self, file_train, number_train, loader=default_loader):
        self.images = file_train
        self.target = number_train
        if 'cifar10' in file_train[0]:
            self.loader = cifar10_default_loader
        elif 'imagenet' in file_train[0]:
            self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img, target

    def __len__(self):
        return len(self.images)
