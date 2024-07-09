import torch.nn as nn
from collections import OrderedDict
from utils.network_utils import get_network
from utils.prune_utils import filter_weights


class ModelBase(object):

    def __init__(self, network, depth, dataset, model=None):
        self._network = network
        self._depth = depth
        self._dataset = dataset
        self.model = model
        self.masks = None
        if self.model is None:
            self.model = get_network(network, depth, dataset)

    def get_ratio_at_each_layer(self):
        assert self.masks is not None, 'Masks should be generated first.'
        res = dict()
        total = 0
        remained = 0
        # for m in self.masks.keys():
        for m in self.model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                mask = self.masks.get(m, None)
                if mask is not None:
                    res[m] = (mask.sum() / mask.numel()).item() * 100
                    total += mask.numel()
                    remained += mask.sum().item()
                else:
                    res[m] = -100.0
                    total += m.weight.numel()
                    remained += m.weight.numel()
        res['ratio'] = remained/total * 100
        return res

    def get_unmasked_weights(self):
        """Return the weights that are unmasked.
        :return dict, key->module, val->list of weights
        """
        assert self.masks is not None, 'Masks should be generated first.'
        res = dict()
        for m in self.masks.keys():
            res[m] = filter_weights(m.weight, self.masks[m])
        return res

    def get_masked_weights(self):
        """Return the weights that are masked.
        :return dict, key->module, val->list of weights
        """
        assert self.masks is not None, 'Masks should be generated first.'
        res = dict()
        for m in self.masks.keys():
            res[m] = filter_weights(m.weight, 1-self.masks[m])
        return res

    def register_mask(self, masks=None):
        # self.masks = None
        self.unregister_mask()
        if masks is not None:
            self.masks = masks
        assert self.masks is not None, 'Masks should be generated first.'
        for m in self.masks.keys():
            m.register_forward_pre_hook(self._forward_pre_hooks)

    def unregister_mask(self):
        for m in self.model.modules():
            m._backward_hooks = OrderedDict()
            m._forward_pre_hooks = OrderedDict()

    def _forward_pre_hooks(self, m, input):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # import pdb; pdb.set_trace()
            mask = self.masks[m]
            m.weight.data.mul_(mask)
        else:
            raise NotImplementedError('Unsupported ' + m)

    def get_name(self):
        return '%s_%s%s' % (self._dataset, self._network, self._depth)

    def train(self):
        self.model = self.model.train()
        return self

    def eval(self):
        self.model = self.model.eval()
        return self

    def cpu(self):
        self.model = self.model.cpu()
        return self

    def cuda(self):
        self.model = self.model.cuda()
        return self

