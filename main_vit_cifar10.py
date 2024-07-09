import os

import argparse
import json
import math
import sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from models.model_base import ModelBase
from models.base.swin import swin_t
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models.base.init_utils import weights_init
from utils.common_utils import (get_logger, makedirs, process_config, PresetLRScheduler, str_to_list)
from utils.data_utils import get_dataloader
from utils.network_utils import get_network
from pruner.GraSP import GraSP
from pruner.SNIP import SNIP
from pruner.SmartRatio import SmartRatio

#scaler = torch.cuda.amp.GradScaler(enabled=True)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--run', type=str, default='')
    args = parser.parse_args()
    runs = None
    if len(args.run) > 0:
        runs = args.run
    config = process_config(args.config, runs)

    return config

def init_logger(config):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path_model = os.path.join(path, 'models/base/%s.py' % config.network.lower())
    path_main = os.path.join(path, 'main_vit_cifar10.py')
    path_pruner = os.path.join(path, 'pruner/%s.py' % config.pruner_file)
    logger = get_logger('log', logpath=config.summary_dir + '/',
                        filepath=path_model, package_files=[path_main, path_pruner])
    logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)
    # sys.stdout = open(os.path.join(config.summary_dir, 'stdout.txt'), 'w+')
    # sys.stderr = open(os.path.join(config.summary_dir, 'stderr.txt'), 'w+')
    return logger, writer


def print_mask_information(mb, logger):
    ratios = mb.get_ratio_at_each_layer()
    logger.info('** Mask information of %s. Overall Remaining: %.2f%%' % (mb.get_name(), ratios['ratio']))
    count = 0
    for k, v in ratios.items():
        if k == 'ratio':
            continue
        logger.info('  (%d) %s: Remaining: %.2f%%' % (count, k, v))
        count += 1


def save_state(net, acc, epoch, loss, config, ckpt_path, is_best=False):
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'loss': loss,
        'args': config
    }
    if not is_best:
        torch.save(state, '%s/pruned_%s_%s%s_%d.t7' % (ckpt_path,
                                                       config.dataset,
                                                       config.network,
                                                       config.depth,
                                                       epoch))
    else:
        torch.save(state, '%s/finetuned_%s_%s%s_best.t7' % (ckpt_path,
                                                            config.dataset,
                                                            config.network,
                                                            config.depth))
def get_exception_layers(net, exception):
    exc = []
    idx = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if idx in exception:
                exc.append(m)
            idx += 1
    return tuple(exc)

def train(net, loader, optimizer, criterion, lr_scheduler, epoch, writer, iteration):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    lr_scheduler(optimizer, epoch)
    desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))

    writer.add_scalar('iter_%d/train/lr' % iteration, lr_scheduler.get_lr(optimizer), epoch)

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
   
        outputs = net(inputs)
        loss = criterion(outputs, targets)
 
        loss.backward()
        optimizer.step()



        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler.get_lr(optimizer), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

    writer.add_scalar('iter_%d/train/loss' % iteration, train_loss / (batch_idx + 1), epoch)
    writer.add_scalar('iter_%d/train/acc' % iteration, 100. * correct / total, epoch)


def test(net, loader, criterion, epoch, writer, iteration):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (0 + 1), 0, correct, total))

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    acc = 100. * correct / total

    writer.add_scalar('iter_%d/test/loss' % iteration, test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('iter_%d/test/acc' % iteration, 100. * correct / total, epoch)
    return acc


def train_once(mb, net, trainloader, testloader, writer, config, ckpt_path, learning_rate,  num_epochs,
               iteration, logger):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    lr_scheduler = PresetLRScheduler(lr_schedule)

    best_acc = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        train(net, trainloader, optimizer, criterion, lr_scheduler, epoch, writer, iteration=iteration)
        test_acc = test(net, testloader, criterion, epoch, writer, iteration)
        lr_schedule.step(epoch)
        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net,
                'acc': test_acc,
                'epoch': epoch,
                'args': config,
                'mask': mb.masks,
                'ratio': mb.get_ratio_at_each_layer()
            }
            path = os.path.join(ckpt_path, 'finetune_%s_%s%s_r%s_it%d_best.pth.tar' % (config.dataset,
                                                                                       config.network,
                                                                                       config.depth,
                                                                                       config.target_ratio,
                                                                                       iteration))
            torch.save(state, path)
            best_acc = test_acc
            best_epoch = epoch
    logger.info('Iteration [%d], best acc: %.4f, epoch: %d' %
                (iteration, best_acc, best_epoch))

def main(config):
    # init logger
    num_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'mnist': 10,
        'tiny_imagenet': 200
    }
    logger, writer = init_logger(config)

    model = swin_t(window_size=4,
                num_classes=10,
                downscaling_factors=(2,2,2,1))

    mask = None
    mb = ModelBase(config.network, config.depth, config.dataset, model)
    mb.cuda()
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(config.size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

    transform_test = transforms.Compose([
        transforms.Resize(config.size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    ckpt_path = config.checkpoint_dir
    num_iterations = config.iterations
    target_ratio = config.target_ratio
    normalize = config.normalize
    # ====================================== fetch exception ======================================
    exception = get_exception_layers(mb.model, str_to_list(config.exception, ',', int))
    logger.info('Exception: ')

    # ====================================== fetch training schemes ======================================
    ratio = 1 - (1 - target_ratio) ** (1.0 / num_iterations)
    learning_rates = str_to_list(config.learning_rate, ',', float)
    #weight_decays = str_to_list(config.weight_decay, ',', float)
    training_epochs = str_to_list(config.epoch, ',', int)
    logger.info('Normalize: %s, Total iteration: %d, Target ratio: %.2f, Iter ratio %.4f.' %
                (normalize, num_iterations, target_ratio, ratio))
    logger.info('Basic Settings: ')
    for idx in range(len(learning_rates)):
        logger.info('  %d: LR: %.5f,  Epochs: %d' % (idx,
                                                              learning_rates[idx],
                                                              #weight_decays[idx],
                                                              training_epochs[idx]))
        
    # ====================================== start pruning ======================================
    iteration = 0
    for _ in range(1):
        logger.info('** Target ratio: %.4f, iter ratio: %.4f, iteration: %d/%d.' % (target_ratio,
                                                                                    ratio,
                                                                                    1,
                                                                                    num_iterations))

        #mb.model.apply(weights_init)
        #print("=> Applying weight initialization(%s)." % config.get('init_method', 'kaiming'))
        print("Iteration of: %d/%d" % (iteration, num_iterations))
        if config.pruner=='SNIP':
            masks = SNIP(mb.model, ratio,trainloader, 'cuda',config,num_classes=num_classes[config.dataset],
                        samples_per_class=config.samples_per_class,
                        num_iters=config.get('num_iters', 1))
        elif config.pruner=='GraSP':
            masks = GraSP(mb.model, ratio,trainloader, 'cuda',config,num_classes=num_classes[config.dataset],
                        samples_per_class=config.samples_per_class,
                        num_iters=config.get('num_iters', 1))
        mb.register_mask(masks)
        # ========== save pruned network ============
        logger.info('Saving..')
        state = {
            'net': mb.model,
            'acc': -1,
            'epoch': -1,
            'args': config,
            'mask': mb.masks,
            'ratio': mb.get_ratio_at_each_layer()
        }
        path = os.path.join(ckpt_path, 'prune_%s_%s%s_r%s_it%d.pth.tar' % (config.dataset,
                                                                           config.network,
                                                                           config.depth,
                                                                           config.target_ratio,
                                                                           iteration))
        torch.save(state, path)

        # ========== print pruning details ============
        logger.info('**[%d] Mask and training setting: ' % iteration)
        print_mask_information(mb, logger)
        logger.info('  LR: %.5f,  Epochs: %d' %
                    (learning_rates[iteration],  training_epochs[iteration]))
        # ========== finetuning =======================
        train_once(mb=mb,
                   net=mb.model,
                   trainloader=trainloader,
                   testloader=testloader,
                   writer=writer,
                   config=config,
                   ckpt_path=ckpt_path,
                   learning_rate=learning_rates[iteration],
                   num_epochs=training_epochs[iteration],
                   iteration=iteration,
                   logger=logger)

    #print(1)


if __name__ == '__main__':
    config = init_config()
    setup_seed(config.seed)
    main(config)
