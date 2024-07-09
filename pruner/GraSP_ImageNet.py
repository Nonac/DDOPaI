import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

from pruner.concepts import concepts,concepts_set


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


def GraSP(net, ratio, train_dataloader, device,config, num_classes=10, samples_per_class=25, num_iters=1):
    eps = 1e-10
    keep_ratio = 1-ratio
    old_net = net

    net = copy.deepcopy(net)
    net.zero_grad()
    weights = []
    total_parameters = count_total_parameters(net)
    fc_parameters = count_fc_parameters(net)

    fc_layers = []
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear):
                fc_layers.append(layer)
            weights.append(layer.weight)
    nn.init.xavier_normal(fc_layers[-1].weight)

    inputs_one = []
    targets_one = []

    grad_w = None
    grad_f = None
    for w in weights:
        w.requires_grad_(True)

    intvs = {
        'cifar10': 128,
        'cifar100': 256,
        'tiny_imagenet': 128,
        'imagenet': 20
    }
    print_once = False

    if config.prune_material=="img":
        dataloader_iter = iter(train_dataloader)
    elif config.prune_material=="concept":
        concept_container=concepts(config.concepts,config.label)
        imgs, y = concept_container.get_concept_patches_fixed_class_concept(
        num_concept=config.pruning_dataset_num_concept
        ,num_class=config.pruning_dataset_num_class
    )
        dataloader_iter = iter(torch.utils.data.DataLoader(
            dataset=concepts_set(imgs, y),
            batch_size=config.batch_size,
            num_workers=16,
            pin_memory=True,
            sampler=None,
            drop_last=False,
            shuffle=True
    ))
    
    for it in range(num_iters):
        print("(1): Iterations %d/%d." % (it, num_iters))
        inputs, targets = next(dataloader_iter)
        N = inputs.shape[0]
        din = copy.deepcopy(inputs)
        dtarget = copy.deepcopy(targets)

        start = 0
        intv = 20

        while start < N:
            end = min(start+intv, N)
            print('(1):  %d -> %d.' % (start, end))
            inputs_one.append(din[start:end])
            targets_one.append(dtarget[start:end])
            outputs = net.forward(inputs[start:end].to(device)) / 200  # divide by temperature to make it uniform
            if print_once:
                x = F.softmax(outputs)
                print(x)
                print(x.max(), x.min())
                print_once = False
            loss = F.cross_entropy(outputs, targets[start:end].to(device))
            grad_w_p = autograd.grad(loss, weights, create_graph=False)
            if grad_w is None:
                grad_w = list(grad_w_p)
            else:
                for idx in range(len(grad_w)):
                    grad_w[idx] += grad_w_p[idx]
            start = end

    for it in range(len(inputs_one)):
        print("(2): Iterations %d/%d." % (it, len(inputs_one)))
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        outputs = net.forward(inputs) / 200  # divide by temperature to make it uniform
        loss = F.cross_entropy(outputs, targets)
        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count] * grad_f[count]).sum()
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

    num_params_to_rm = int(len(all_scores) * (1 - keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)
    keep_masks = dict()
    for m, g in grads.items():
        keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return keep_masks





