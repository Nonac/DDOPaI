import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import copy
import types
import math

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


def SmartRatio(net, ratio, device,args):
    keep_ratio = 1-ratio
    old_net=net
    net = copy.deepcopy(net)  # .eval()
    net.zero_grad()

    total_parameters = count_total_parameters(net)
    fc_parameters = count_fc_parameters(net)


    # ========== the following code is the implementation of our smart ratio ============
    # ========== default = 0.3 ============
    linear_keep_ratio = args.linear_keep_ratio

    # ========== calculate the sparsity using order statistics ============
    CNT = 0
    Num = []
    # ========== calculate the number of layers and the corresponding number of weights ============
    for idx, m in enumerate(net.modules()):
        if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
            Num.append(m.weight.data.view(-1).size()[0])
            CNT = CNT + 1
                
    Num = torch.from_numpy(np.array(Num)).float()
        
    # ========== set ratio ============
    n = CNT
    Ratio = torch.rand(1,CNT)
    for i in range(CNT):
        k = i + 1 # 1~CNT
        Ratio[0][n-k] = (k)**2 + k
        if args.linear_decay != 0:
            Ratio[0][n-k] = k
        if args.ascend != 0:
            Ratio[0][n-k] = (n-k+1)**2 + (n-k+1)
        if args.cubic != 0:
            Ratio[0][n-k] = (k)**3
    #if args.pruner=='AR':
     #   for i in range(CNT):
     #       k = i + 1 # 1~CNT
      #      Ratio[0][n-k] = ratio

    Ratio = Ratio[0]

    num_now = 0
    total_num = 0
    linear_num = 0
    
    
    # ========== calculation and scaling ============
    i = 0
    TEST = 0
    for m in net.modules():
        if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
            if not isinstance(m,nn.Linear):
                num_now = num_now + int((Ratio[i])*Num[i])
                if args.arch != 'resnet':
                    TEST = TEST + int(Num[i]*Ratio[i]/(i+1)**2)
                else:
                    TEST = TEST + int(Num[i]*Ratio[i])
            else:
                linear_num = linear_num + Num[i]
            total_num = total_num + Num[i]
            i = i + 1

    goal_num = int(total_num * (1-ratio)) - int(linear_num*linear_keep_ratio)
    # ========== since the #linear_num is much lesser than that of total_num ============
    # ========== one can just easily set balance_ratio = 1 - init_prune_ratio without hurting the performance ============
    balance_ratio = goal_num / (total_num - linear_num)
    # TEST
    k = (goal_num) / TEST
    i = 0
    for m in net.modules():
        if isinstance(m,nn.Conv2d):
            if args.arch != 'resnet':
                if args.pruner=='AR':
                    Ratio[i]=balance_ratio
                else:
                    Ratio[i] = Ratio[i] * k / (i+1)**2
            else:
                if args.pruner=='AR':
                    Ratio[i]=balance_ratio
                else:
                    Ratio[i] = Ratio[i] * k
            i = i + 1     
    
    
    # ========== if the prune-ratio is too small, then some keep_ratio will > 1 ============
    # ========== the easy modification ============
    if args.pruner!='AR':
        ExtraNum = 0
        i = 0
        for m in net.modules():
            size = Num[i]
            if isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d):
                if not isinstance(m,nn.Linear):
                    if Ratio[i] >= 1:
                        ExtraNum = ExtraNum + int((Ratio[i]-1)*size)
                        Ratio[i] = 1
                    else:
                        RestNum = int((1-Ratio[i])*Num[i])
                        if RestNum >= ExtraNum:
                            Ratio[i] = Ratio[i] + ExtraNum/Num[i]
                            ExtraNum = 0
                        else:
                            ExtraNum = ExtraNum - RestNum
                            Ratio[i] = 1
                if ExtraNum == 0:
                    break
                i = i + 1
    
    # ========== set the smart-ratio masks ============
    #keep_masks = []
    CNT = 0
    keep_masks = dict()

    for m in old_net.modules():
        if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
            mask = m.weight.data.abs().clone().float().cuda()
            Size = mask.size()
            mask = mask.view(-1)
            keep_ratio = Ratio[CNT]
            num_keep = int((keep_ratio)*Num[CNT])
            if Ratio[CNT] >= 1:
                num_keep = int(Num[CNT])
            if args.uniform != 0:
                Ratio[CNT] = balance_ratio
                num_keep = int(Ratio[CNT]*Num[CNT])
            if isinstance(m,nn.Linear):
                num_keep = int(linear_keep_ratio*Num[CNT])
            # ========== this judgement is for our hybrid ticket ============
            # ========== if specify the hybrid method, our smart ratio will combine the magnitude-based pruning ============
            if args.hybrid != 0:
                print("################### DEBUG PRINT : USING HYBRID TICKET ###################")
                value,idx = torch.topk(mask,num_keep)
                temp = torch.zeros(int(Num[CNT]))
                temp[idx] = 1.0
                mask = temp.clone().float().cuda()
            
            else:
                temp = torch.ones(1,num_keep)
                mask[0:num_keep] = temp
                temp = torch.zeros(1,int(Num[CNT].item()-num_keep))
                mask[num_keep:] = temp
                mask = mask.view(-1)[torch.randperm(mask.nelement())].view(mask.size())
            
            
            
            CNT = CNT + 1
            keep_masks[m]=mask.view(Size)
            
    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return keep_masks