import argparse
import os
import torch
import torch.nn as nn

from models.model_base import ModelBase
from tensorboardX import SummaryWriter
from models.base.init_utils import weights_init
from utils.common_utils import (get_logger, makedirs, process_config, str_to_list)
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
from pruner.SmartRatio import SmartRatio
from pruner.GraSP_ImageNet import GraSP
from pruner.SNIP import SNIP


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
    path_model = os.path.join(path, 'models/base/%s.py' % 'vgg')
    path_main = os.path.join(path, 'main_prune_imagenet.py')
    path_pruner = os.path.join(path, 'pruner/%s.py' % config.pruner_file)
    logger = get_logger('log', logpath=config.summary_dir+'/',
                        filepath=path_model, package_files=[path_main, path_pruner])
    logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)
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


def get_exception_layers(net, exception):
    exc = []
    idx = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if idx in exception:
                exc.append(m)
            idx += 1
    return tuple(exc)


def main(config):
    # init logger
    classes = {
        'cifar10': 10,
        'cifar100': 100,
        'mnist': 10,
        'tiny_imagenet': 200,
        'imagenet': 1000
    }
    logger, writer = init_logger(config)

    # build model
    model = models.__dict__[config.network]()
    mb = ModelBase(config.network, config.depth, config.dataset, model)
    mb.cuda()

    # preprocessing
    # ====================================== fetch configs ======================================
    ckpt_path = config.checkpoint_dir
    num_iterations = config.iterations
    target_ratio = config.target_ratio
    normalize = config.normalize
    # ====================================== fetch exception ======================================
    exception = get_exception_layers(mb.model, str_to_list(config.exception, ',', int))
    logger.info('Exception: ')

    for idx, m in enumerate(exception):
        logger.info('  (%d) %s' % (idx, m))

    # ====================================== fetch training schemes ======================================
    ratio = 1-(1-target_ratio) ** (1.0 / num_iterations)
    learning_rates = str_to_list(config.learning_rate, ',', float)
    weight_decays = str_to_list(config.weight_decay, ',', float)
    training_epochs = str_to_list(config.epoch, ',', int)
    logger.info('Normalize: %s, Total iteration: %d, Target ratio: %.2f, Iter ratio %.4f.' %
                (normalize, num_iterations, target_ratio, ratio))
    logger.info('Basic Settings: ')
    for idx in range(len(learning_rates)):
        logger.info('  %d: LR: %.5f, WD: %.5f, Epochs: %d' % (idx,
                                                              learning_rates[idx],
                                                              weight_decays[idx],
                                                              training_epochs[idx]))


    # ====================================== get dataloader ======================================
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        config.traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=250, shuffle=True,
        num_workers=16, pin_memory=True, sampler=None)

    # ====================================== start pruning ======================================

    for iteration in range(num_iterations):
        logger.info('** Target ratio: %.4f, iter ratio: %.4f, iteration: %d/%d.' % (target_ratio,
                                                                                    ratio,
                                                                                    iteration,
                                                                                    num_iterations))

        assert num_iterations == 1
        print("=> Applying weight initialization.")
        mb.model.apply(weights_init)

        if config.pruner=='GraSP':
            masks = GraSP(mb.model, ratio, trainloader, 'cuda',config,
                        num_classes=classes[config.dataset],
                        samples_per_class=config.samples_per_class,
                        num_iters=config.get('num_iters', 1))
        elif config.pruner=='SNIP':
            masks = SNIP(mb.model, ratio, trainloader, 'cuda',config,
                        num_classes=classes[config.dataset],
                        samples_per_class=config.samples_per_class,
                        num_iters=config.get('num_iters', 1))
        elif config.pruner=='SR':
            masks=SmartRatio(mb.model,ratio,'cuda',config)

        # ========== register mask ==================
        mb.masks = masks
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


if __name__ == '__main__':
    config = init_config()
    main(config)
