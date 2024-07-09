import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math

import copy
import types

from utils.data_utils import get_dataloader
from utils.common_utils import (get_logger, makedirs, process_config, str_to_list)
#from concept_for_Tiny_Imagenet import concepts,concepts_set,get_tiny_imagenet_condept_dataloader
from pruner.concepts import concepts,concepts_set


def get_exception_layers(net, exception):
    exc = []
    idx = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if idx in exception:
                exc.append(m)
            idx += 1
    return tuple(exc)


def SNIP_fetch_data(dataloader,batch_size, num_classes=1000, samples_per_class=1):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break
    xs=[]
    ys=[]
    minibatch=[]
    minitarget=[]
    for x,y in zip(datas,labels):
        if len(minibatch)<batch_size:
            minibatch.append(torch.cat(x,0))
            minitarget.append(torch.cat(y))
        else:
            xs.append(torch.cat(minibatch))
            ys.append(torch.cat(minitarget).view(-1))
            minibatch = []
            minitarget = []
    xs.append(torch.cat(minibatch))
    ys.append(torch.cat(minitarget).view(-1))

    return xs, ys


def count_total_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
    return total


def count_fc_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear)):
            total += m.weight.numel()
    return total

def snip_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)


def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)

def SNIP(net, ratio, train_dataloader, device,config, num_classes=10, samples_per_class=25, num_iters=1):
    eps = 1e-10
    keep_ratio = 1 - ratio
    old_net = net

    net = copy.deepcopy(net)  # .eval()


    exception = get_exception_layers(net, str_to_list(config.exception, ',', int))
    net.zero_grad()

    weights = []
    total_parameters = count_total_parameters(net)
    fc_parameters = count_fc_parameters(net)

    #rescale_weights(net)
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            weights.append(layer.weight)
        # 检查是否是 Linear 层并且有 bias
        #elif isinstance(layer, nn.Linear) and layer.bias is not None:
        elif isinstance(layer, nn.Linear):
            weights.append(layer.weight)


    inputs_one = []
    targets_one = []

    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    print_once = False


    if config.prune_material=='concept':
        concept_container=concepts(config.concepts,config.label)
        imgs, y = concept_container._get_concept_patches_fixed_class_concept(
        num_concept=config.pruning_dataset_num_concept
        ,num_class=config.pruning_dataset_num_class)
        dataloader_iter = iter(torch.utils.data.DataLoader(
            dataset=concepts_set(imgs, y),
            batch_size=config.batch_size,
            num_workers=16,
            pin_memory=True,
            sampler=None,
            drop_last=False,
            shuffle=True
    ))

    
    if config.prune_material=="concept" or config.prune_material=="ss":
        for it in range(len(dataloader_iter)):
            print("Iterations %d/%d." % (it+1, len(dataloader_iter)))
            inputs, targets = next(dataloader_iter)
            N = inputs.shape[0]
            din = copy.deepcopy(inputs)
            dtarget = copy.deepcopy(targets)
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = net.forward(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
    elif config.prune_material=="img":
        
        if "brightness_ablation" in config:
            if config.brightness_ablation:
                train_dataloader, _ = get_dataloader('tiny_imagenet_ablation', config.batch_size, 256, 4,config=config)
        
        for it in range(num_iters):
         
            examples,labels= SNIP_fetch_data(train_dataloader,200, num_classes, samples_per_class) 

            for inputs, targets in zip(examples,labels):
                N = inputs.shape[0]
                din = copy.deepcopy(inputs)
                dtarget = copy.deepcopy(targets)
                inputs = inputs.to(device)
                # all one metrix ablation experiment
                #inputs= torch.ones(inputs.shape).to(device)
                targets = targets.to(device)
                outputs = net.forward(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
            print(f"The {(it+1)*num_classes} samples have pruned")
                

    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
       # if isinstance(layer, nn.Conv2d) or (isinstance(layer, nn.Linear) and layer.bias is not None):
        if isinstance(layer, nn.Conv2d) or (isinstance(layer, nn.Linear)):
            if layer in exception:
                pass
            else:
                grads[old_modules[idx]] = abs(layer.weight.data * layer.weight.grad)  # -theta_q g

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_kp = int(len(all_scores) * (keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_kp, sorted=True)

    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)
    keep_masks = dict()
    for m, g in grads.items():
        keep_masks[m] = ((g / norm_factor) >= acceptable_score).float()

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return keep_masks