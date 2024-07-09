import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math

import copy
import types

from utils.data_utils import get_dataloader
#from pruner.concept_for_Tiny_Imagenet import concepts,get_tiny_imagenet_condept_dataloader
from pruner.concepts import concepts,concepts_set

def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
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

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y


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


def GraSP(net, ratio, train_dataloader, device,config, num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True):
    #DD
    if config.prune_material == 'concept':
        concept=concepts(config.concepts,config.label)
        imgs,y=concept.get_concept_patches_fixed_class_concept(config.num_concept,config.num_classes)
        concept_dataloader=  iter(torch.utils.data.DataLoader(
            dataset=concepts_set(imgs, y),
            batch_size=config.batch_size,
            num_workers=16,
            pin_memory=True,
            sampler=None,
            drop_last=False,
            shuffle=True
    ))
    
    #Super stitching
    if config.prune_material == 'ss':
        concept=concepts(config.stitching,config.label)
        imgs,y=concept.get_stitching_patches(config)
        concept_dataloader=  iter(torch.utils.data.DataLoader(
            dataset=concepts_set(imgs, y),
            batch_size=config.batch_size,
            num_workers=16,
            pin_memory=True,
            sampler=None,
            drop_last=False,
            shuffle=True
    ))
    
    eps = 1e-10
    keep_ratio = 1-ratio
    old_net = net

    net = copy.deepcopy(net)  # .eval()
    net.zero_grad()

    weights = []
    total_parameters = count_total_parameters(net)
    fc_parameters = count_fc_parameters(net)

    # rescale_weights(net)
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal(layer.weight)
            weights.append(layer.weight)

    inputs_one = []
    targets_one = []

    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    print_once = False
    if "brightness_ablation" in config:
        if config.brightness_ablation:
            train_dataloader, _ = get_dataloader('tiny_imagenet_ablation', config.batch_size, 256, 4,config=config)
    for it in range(num_iters):
        print("(1): Iterations %d/%d." % (it, num_iters))
        if config.prune_material == 'ss' or config.prune_material == 'concept':
            inputs, targets = next(concept_dataloader)
            #inputs, targets = GraSP_fetch_data(concept_dataloader, num_classes, samples_per_class)
        else:   
            inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
        N = inputs.shape[0]
        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)
        inputs_one.append(din[:N//2])
        targets_one.append(dtarget[:N//2])
        inputs_one.append(din[N // 2:])
        targets_one.append(dtarget[N // 2:])
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net.forward(inputs[:N//2])/T
        if print_once:
            # import pdb; pdb.set_trace()
            x = F.softmax(outputs)
            print(x)
            print(x.max(), x.min())
            print_once = False
        loss = F.cross_entropy(outputs, targets[:N//2])
        # ===== debug ================
        grad_w_p = autograd.grad(loss, weights)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

        outputs = net.forward(inputs[N // 2:])/T
        loss = F.cross_entropy(outputs, targets[N // 2:])
        grad_w_p = autograd.grad(loss, weights, create_graph=False)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

    ret_inputs = []
    ret_targets = []
    
    

    for it in range(len(inputs_one)):
        print("(2): Iterations %d/%d." % (it, num_iters))
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        ret_inputs.append(inputs)
        ret_targets.append(targets)
        outputs = net.forward(inputs)/T
        loss = F.cross_entropy(outputs, targets)
        # ===== debug ==============

        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads[old_modules[idx]] = -layer.weight.data * layer.weight.grad  # -theta_q Hg

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)
    keep_masks = dict()
    for m, g in grads.items():
        keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()
        if 'random_prune' in config:
            if config.random_prune:
                keep_masks[m] = keep_masks[m].view(-1)[torch.randperm(keep_masks[m].nelement())].view(keep_masks[m].size())
                print("random prune")

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return keep_masks
