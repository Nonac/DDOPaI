from models.base import (VGG,
                         resnet)
import torch
from models.base import wide_resnet18,wide_resnet34,wide_resnet50,wide_resnet101,wide_resnet152
from models.base import mobilenet_v2

WIDE_RESNET={
    18: wide_resnet18,
    34: wide_resnet34,
    50: wide_resnet50,
    101: wide_resnet101,
    152: wide_resnet152
}


def get_network(network, depth, dataset, use_bn=True):
    if network == 'vgg':
        print('Use batch norm is: %s' % use_bn)
        return VGG(depth=depth, dataset=dataset, batchnorm=use_bn)
    elif network == 'resnet':
        return resnet(depth=depth, dataset=dataset)
    elif network == 'wide_resnet':
        return WIDE_RESNET[depth]((3, 64, 64), 200, dense_classifier=False, pretrained=False)
    elif network == 'mobilenet_v2':
        return mobilenet_v2()
    else:
        raise NotImplementedError('Network unsupported ' + network)


def stablize_bn(net, trainloader, device='cuda'):
    """Iterate over the dataset for stabilizing the
    BatchNorm statistics.
    """
    net = net.train()
    for batch, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        net(inputs)