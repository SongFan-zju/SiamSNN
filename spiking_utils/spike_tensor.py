# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /SiamSNN/spiking_utils/spike_tensor.py
# Date: 2021-12-14
# Author: FanJH
# Description:
#############################################
from matplotlib.pyplot import sca
import matplotlib.pyplot as plt

from numpy.random.mtrand import binomial
import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.binomial import Binomial
import numpy as np

# from spiking_utils.spike_layer import SpikeLinear

firing_ratio_record = False
firing_ratios = []


class SpikeTensor:

    def __init__(self, data, timesteps, scale_factor):
        """
        data shape: [t*batch, channel, height, width]
        """
        self.data = data
        self.timesteps = timesteps
        self.b = self.data.size(0) // timesteps
        self.chw = self.data.size()[1:]
        # self.scale_factor = torch.ones([*self.data.size()[1:]]).to(data.device)
        if not hasattr(scale_factor, "shape"):
            self.scale_factor = torch.ones(self.data.size(1)).to(data.device)
        else:
            self.scale_factor = torch.ones(scale_factor.shape).to(data.device)
        if isinstance(scale_factor, torch.Tensor):
            dim = scale_factor.dim()
            # if dim == 1:
            #     self.scale_factor *= scale_factor.view(-1, *([1] * (len(self.chw) - 1)))
            # else:
            self.scale_factor *= scale_factor
        else:
            self.scale_factor.fill_(scale_factor)
        if firing_ratio_record:
            firing_ratios.append(self.firing_ratio())

    def firing_ratio(self):
        firing_ratio = torch.mean(self.data.view(self.timesteps, -1, *self.chw), 0)
        return firing_ratio

    def timestep_dim_tensor(self):
        return self.data.view(self.timesteps, -1, *self.chw)

    def size(self, *args):
        return self.data.size(*args)

    def view(self, *args):
        return SpikeTensor(self.data.view(*args), self.timesteps, self.scale_factor)

    def to_float(self):
        assert self.scale_factor is not None
        firing_ratio = self.firing_ratio()
        scaled_float_tensor = firing_ratio * self.scale_factor[None, :, None, None]  #.unsqueeze(0)
        return scaled_float_tensor

    def __str__(self):
        return f"Spiketensor T{self.timesteps} Shape({self.b} {self.chw}) ScaleFactor {self.scale_factor} \n{self.data.shape}"


def plt_spike_counts(cnts, labels):

    ori = cnts[0]
    xticks = range(len(ori))

    x = np.arange(len(xticks))  # the label locations
    width = 0.6  # the width of the bars

    fig, ax = plt.subplots()
    w = width / len(cnts)
    for i, cnt in enumerate(cnts):
        bar = ax.bar(x + w * i, cnt, w, label=labels[i])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('spikes counts')
    ax.set_title('spikes counts')
    ax.set_xticks(x, xticks)
    ax.legend()

    fig.tight_layout()

    plt.show()


def GetRateSpikeTensor(tensor, timesteps):
    # dist = Bernoulli(tensor)
    # spike_tensor = dist.sample((timesteps,))
    '''
    spike_num = torch.ceil(tensor*timesteps)
    # num = np.random.choice(range(0,timesteps),spike_num,replace=False)
    spike_tensor = torch.cat([spike_num.unsqueeze(0)-ind for ind in range(timesteps)],0)
    spike_tensor = torch.where(spike_tensor>0,torch.tensor(1,dtype=torch.float32,device=tensor.device),
                                                torch.tensor(0,dtype=torch.float32,device=tensor.device))
    spike_tensor = torch.reshape(spike_tensor,[-1,*tensor.size()[1:]])
    '''

    spike_tensor_norm = GetRateNormSpikeTensor(tensor, timesteps)
    # spike_tensor_norm = GetRateSpikeTensor_new(tensor, timesteps)

    # spike_tensor_sparse = GetRateSparseSpikeTensor(tensor, timesteps)
    # snc = GetRateNormClipSpikeTensor(tensor, timesteps)
    #
    # sc = torch.sum(spike_tensor, dim=(1, 2, 3))
    # scn = torch.sum(spike_tensor_norm, dim=(1, 2, 3))
    # scs = torch.sum(spike_tensor_sparse, dim=(1, 2, 3))
    # scnc = torch.sum(snc, dim=(1, 2, 3))

    # plt_spike_counts([sc.cpu(), scn.cpu(), scs.cpu(), scnc.cpu()], ['origin', 'norm', 'sparse', 'norm_clip'])

    return spike_tensor_norm


def GetRateNormSpikeTensor(tensor, timesteps):
    spike_num = torch.ceil(tensor * timesteps)
    # normalized to [1, max - min + 1]
    spike_num = spike_num - torch.min(spike_num) + 1

    # num = np.random.choice(range(0,timesteps),spike_num,replace=False)
    spike_tensor = torch.cat([spike_num.unsqueeze(0) - ind * 255 for ind in range(timesteps)], 0)
    spike_tensor = torch.where(spike_tensor > 0, torch.tensor(1, dtype=torch.float32, device=tensor.device),
                               torch.tensor(0, dtype=torch.float32, device=tensor.device))
    spike_tensor = torch.reshape(spike_tensor, [-1, *tensor.size()[1:]])

    return spike_tensor


def GetRateSpikeTensor_new(tensor, timesteps):
    # with normalization
    # th_tensor = torch.rand((timesteps, *tensor.shape)).to(tensor.device)
    # th_tensor[:] = 0.4
    # probs = (torch.abs(tensor / 255)).repeat(timesteps, 1, 1, 1, 1)

    # without normalization
    th_tensor = torch.ones((timesteps, *tensor.shape)).to(tensor.device)
    th_tensor *= 100  #(tensor.median()).int()
    probs = tensor.repeat(timesteps, 1, 1, 1, 1)

    spike_tensor = torch.where(probs > th_tensor, torch.tensor(1, dtype=torch.float32, device=tensor.device),
                               torch.tensor(0, dtype=torch.float32, device=tensor.device))
    spike_tensor = torch.reshape(spike_tensor, [-1, *tensor.size()[1:]])
    return spike_tensor


def GetRateNormClipSpikeTensor(tensor, timesteps):
    spike_num = torch.ceil(tensor * timesteps)
    # normalized to [1, max - min + 1]
    spike_num = spike_num - torch.min(spike_num) - 1

    # num = np.random.choice(range(0,timesteps),spike_num,replace=False)
    spike_tensor = torch.cat([spike_num.unsqueeze(0) - ind for ind in range(timesteps)], 0)
    spike_tensor = torch.where(spike_tensor > 0, torch.tensor(1, dtype=torch.float32, device=tensor.device),
                               torch.tensor(0, dtype=torch.float32, device=tensor.device))
    spike_tensor = torch.reshape(spike_tensor, [-1, *tensor.size()[1:]])

    return spike_tensor


def GetRateSparseSpikeTensor(tensor, timesteps):
    # time shift + spikes unbiased
    spike_num = torch.ceil(tensor * timesteps)
    m = torch.min(spike_num)
    sn_bias = spike_num - m

    st = torch.cat([sn_bias.unsqueeze(0) * 2 - ind for ind in range(timesteps)], 0)
    snr = sn_bias.repeat(timesteps, 1, 1, 1, 1)

    st = torch.where((st <= snr) & (st >= 0), torch.tensor(1, dtype=torch.float32, device=tensor.device),
                     torch.tensor(0, dtype=torch.float32, device=tensor.device))

    # # num = np.random.choice(range(0,timesteps),spike_num,replace=False)
    # spike_tensor = torch.cat([spike_num.unsqueeze(0)-ind for ind in range(timesteps)],0)

    # spike_tensor = torch.where(spike_tensor>0,torch.tensor(1,dtype=torch.float32,device=tensor.device),
    #                                             torch.tensor(0,dtype=torch.float32,device=tensor.device))
    st = torch.reshape(st, [-1, *tensor.size()[1:]])

    return st
