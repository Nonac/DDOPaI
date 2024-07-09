import sys
import os
import numpy as np
import copy
import shutil
import sklearn.metrics as metrics
import torchvision
import abc
import ACE.pytorch.ace_helpers as ace_helpers
from ACE.pytorch.ace import ConceptDiscovery, make_model

import argparse
import torch
import json
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor



class ACE(abc.ABC):
    def __init__(self,config, model,samples=None,target=None,labels=None,dataloader=None):
        self.config = config
        self.model = model
        self.samples = samples
        self.target = target
        self.labels = labels
        self.dataloader=dataloader
        
        
    def _pre_process_dataloader(self,max_imgs,num_classes):
        if self.dataloader is None:
            dataloader= torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(
                    self.samples, self.target
                    ), batch_size=1)
        else:
            dataloader=self.dataloader
        img=[[] for _ in range(num_classes)]
        for i, (input, target) in enumerate(dataloader):
            #if len(img[torch.argmax(target)])<max_imgs:
                #img[torch.argmax(target)].append(input.permute(0, 2, 3, 1)[0].numpy())
            if len(img[target[0]])<max_imgs:
                img[target[0]].append(input.permute(0, 2, 3, 1)[0].numpy())
            else:
                continue
            if all(len(img[j]) == max_imgs for j in range(num_classes)):
                break
        return [np.array(img_array) for img_array in img]
    
    
    def concept_discovery(self):
        # Open txt file
        label_list = []
        with open(self.config.LABEL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                label_list.append(line.strip())
        # Make model
        self.model=make_model(self.config,label_list,self.model)
        #DEBUG

        
        self.imgs=self._pre_process_dataloader(self.config.MAX_IMGS,len(label_list))
        self.tcav_score_imgs=copy.deepcopy(self.imgs)
        self.ConceptPatch={}
        if os.path.exists(self.config.WORKING_DIR):
            shutil.rmtree(self.config.WORKING_DIR)
        os.makedirs(self.config.WORKING_DIR)
        
      
        
        for idx,target_class in enumerate(label_list): 
            discovered_concepts_dir = os.path.join(self.config.WORKING_DIR,target_class, 'concepts/')
            results_dir = os.path.join(self.config.WORKING_DIR,target_class, 'results/')
            self.cavs_dir = os.path.join(self.config.WORKING_DIR,target_class, 'cavs/')
            self.activations_dir = os.path.join(self.config.WORKING_DIR,target_class, 'acts/')
            results_summaries_dir = os.path.join(self.config.WORKING_DIR,target_class, 'results_summaries/')
            os.makedirs(discovered_concepts_dir)
            os.makedirs(results_dir)
            os.makedirs(self.cavs_dir)
            os.makedirs(self.activations_dir)
            os.makedirs(results_summaries_dir)
            
            if self.imgs[idx].shape[0]==0:
                self.ConceptPatch[target_class]={
                    self.config.FEATHER_NAMES[0]:{}
                }
                continue
            print("Starting {} concept discovery".format(target_class))
            cd = ConceptDiscovery(
                self.model,
                target_class,
                self.config.RANDOM_CONCEPT,
                self.config.FEATHER_NAMES,
                "",
                self.activations_dir,
                self.cavs_dir,
                num_random_exp=self.config.NUM_RANDOM_EXP,
                channel_mean=True,
                max_imgs=self.config.MAX_IMGS,
                min_imgs=self.config.MIN_IMGS,
                num_discovery_imgs=self.config.MAX_IMGS,
                mean=self.config.MEAN,
                std=self.config.STD,
                data=self.imgs[idx],
                RANDOM_SOURCE=self.config.RANDOM_SOURCE
            )
            cd.image_shape=self.config.IMAGE_SHAPE
            cd.create_patches(
                discovery_images=cd.data,
                param_dict={"n_segements":self.config.N_SEGMENTS}
                )
            print('Saving the concept discovery target class images')
            image_dir = os.path.join(discovered_concepts_dir, 'images')
            os.makedirs(image_dir)
            ace_helpers.save_images(image_dir,
                            (cd.discovery_images * 256).astype(np.uint8))
            cd.discover_concepts(method='KM', param_dicts={'n_clusters': 25})
            ace_helpers.save_concepts(cd, discovered_concepts_dir)
            
            cav_accuraciess = cd.cavs(min_acc=0.0)
            scores = cd.tcavs(test=True,tcav_score_images=self.tcav_score_imgs[idx])
            
            ace_helpers.save_ace_report(cd, cav_accuraciess, scores,
                                 results_summaries_dir)
            # Plot examples of discovered concepts
            for bn in cd.bottlenecks:
                ace_helpers.plot_concepts(cd, bn, 10, address=results_dir)
            # Delete concepts that don't pass statistical testing
            print('plotting')
            cd.test_and_remove_concepts(scores)
            self.ConceptPatch[target_class]=ace_helpers.return_concepts(cd,scores)
            #del cd

        return self.ConceptPatch
        

                