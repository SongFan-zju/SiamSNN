from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import cv2
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, MultiStepLR

from got10k.trackers import Tracker

from spiking_utils.spike_tensor import SpikeTensor, GetRateSpikeTensor, GetRateSpikeTensor_new
import pickle
import pandas as pd
import os
import shutil


class SiamFC(nn.Module):

    def __init__(self):
        super(SiamFC, self).__init__()
        # nn.BatchNorm2d(8, eps=1e-6, momentum=0.05),
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=4, bias=False),
            nn.ReLU(inplace=True),

            # conv2
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, bias=False),
            nn.ReLU(inplace=True),

            # conv3
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, bias=False))
        self._initialize_weights()

    def forward(self, z, x):
        z = self.feature(z)
        x = self.feature(x)

        # fast cross correlation
        n, c, h, w = x.size()
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))

        # adjust the scale of responses
        out = 0.001 * out + 0.0

        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight.data, mode='fan_out',
                #                      nonlinearity='relu')
                init.xavier_normal_(m.weight.data, gain=1)
                # m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class SiamFCLarger(SiamFC):

    def __init__(self):
        super(SiamFCLarger, self).__init__()
        # nn.BatchNorm2d(8, eps=1e-6, momentum=0.05),
        self.feature = nn.Sequential(

            # conv1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, bias=False),
            nn.ReLU(inplace=True),

            # conv2
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, bias=False),
            nn.ReLU(inplace=True),
            # conv3
            # nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, bias=False),
            # nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, bias=False))
        self._initialize_weights()


class TrackerSiamFC(Tracker):

    def __init__(self, normalize, epoch_num, net_path=None, **kargs):
        super(TrackerSiamFC, self).__init__(name='SiamFC', is_deterministic=True)
        self.cfg = self.parse_args(**kargs)
        self.normalize = normalize
        self.epoch_num = epoch_num

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = SiamFC()
        if net_path is not None:
            self.net.load_state_dict(torch.load(net_path, map_location=lambda storage, loc: storage))
            self.net = self.net.to(self.device)
        self.snn = None
        # setup optimizer
        # self.optimizer = optim.SGD(
        #     self.net.parameters(),
        #     lr=self.cfg.initial_lr,
        #     weight_decay=self.cfg.weight_decay,
        #     momentum=self.cfg.momentum)
        self.optimizer = optim.AdamW(self.net.parameters(), lr=self.cfg.initial_lr, weight_decay=0.0005)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch_num, eta_min=1e-6)
        # setup lr scheduler
        # self.lr_scheduler = ExponentialLR(
        #     self.optimizer, gamma=self.cfg.lr_decay)
        # self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[50, 100], gamma=0.1)

        # self.lr_scheduler = ReduceLROnPlateau(
        #     self.optimizer, 'min', factor=0.5, patience=15, verbose=True)

    def parse_args(self, **kargs):
        # default parameters
        cfg = {
            # inference parameters
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            'scale_num': 1,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 9,
            'response_up': 16,
            'total_stride': 8,
            'adjust_scale': 0.001,
            # train parameters
            'initial_lr': 0.01,  #0.01,
            'lr_decay': 0.8685113737513527,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0
        }

        # for key, val in kargs.items():
        #     if key in cfg:
        #         cfg.update({key: val})

        return namedtuple('GenericDict', cfg.keys())(**cfg)

    def save_csv(self, tensor: SpikeTensor, video, frame_id):
        if frame_id > 10000:
            return
        data = tensor.data
        video = str(video)
        spike = torch.nonzero(data)
        first_col = spike[:, 0]
        unique_vals = torch.unique(first_col)

        save_dir = os.path.join("spike", video, str(frame_id))
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

        for val in unique_vals:
            mask = first_col == val
            group_tensor = spike[mask].cpu().numpy()
            save_path = os.path.join("spike", video, f"{frame_id}", f"{val}.csv")
            # 这里每个 npz 只保存一个数组，命名为 data
            # np.savez_compressed(save_path, data=group_tensor)
            df = pd.DataFrame(group_tensor)
            df.to_csv(save_path, index=False, header=False)

    def save_out(self, tensor: SpikeTensor, conv, frame_id):
        lst = [1, 4, 7]
        if conv not in lst:
            return
        if frame_id > 10000:
            return
        data = tensor.data
        if conv // 3 + 1 == 3:
            save_path = os.path.join("pcss_output", f"pcss_out_{frame_id}.csv".format(conv // 3 + 1))
            spike = torch.nonzero(data)
            df = pd.DataFrame(spike.cpu().numpy())
            df.to_csv(save_path, index=False, header=False)

    def init(self, image, box, args, frame_id, from_fpga=None, video_name="video1", save_csv=False):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.normalize:
            image = np.asarray(image[..., np.newaxis]) / 255.0  #gray image normalization
        else:
            image = np.asarray(image[..., np.newaxis])

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([box[1] - 1 + (box[3] - 1) / 2, box[0] - 1 + (box[2] - 1) / 2, box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(np.hanning(self.upscale_sz), np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step**np.linspace(-(self.cfg.scale_num // 2), self.cfg.scale_num // 2,
                                                              self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(image,
                                               self.center,
                                               self.z_sz,
                                               out_size=self.cfg.exemplar_sz,
                                               pad_color=self.avg_color)

        # exemplar features
        exemplar_image = torch.from_numpy(exemplar_image).to(self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            if args.nntype == "ann":
                self.net.eval()
                self.kernel = self.net.feature(exemplar_image)
            elif args.nntype == "snn":
                assert self.snn is not None, "snn model not builded."
                self.snn.eval()
                spike_image_tensor = GetRateSpikeTensor(exemplar_image, args.timesteps)
                # spike_image_tensor = GetRateSpikeTensor_new(exemplar_image,args.timesteps)
                size = spike_image_tensor.shape[2]
                for i in range(3, size, 4):
                    spike_image_tensor[:, :, i, :] = 0
                    spike_image_tensor[:, :, :, i] = 0
                spike_exemplar_image = SpikeTensor(spike_image_tensor, args.timesteps, scale_factor=1.)
                if save_csv:
                    self.save_csv(spike_exemplar_image, video=video_name, frame_id=frame_id)
                out = spike_exemplar_image

                i = 1
                for layer in self.snn.children():
                    out = layer(out)
                    self.save_out(out, i, frame_id)
                    i += 1
                # out = self.snn(spike_exemplar_image)
                if from_fpga is not None:
                    tmp_data = torch.from_numpy(from_fpga).unsqueeze(0).to("cuda:0").float()
                    if tmp_data.sum() != out.data.sum():
                        print(tmp_data.sum())
                        print(out.data.sum())
                        print(frame_id)
                    out.data = tmp_data
                self.kernel = out.to_float()

                # f1 = open(f'./snn_debug/fpga/exemplar_snn_simple_{frame_id}.data', 'wb')
                # pickle.dump(self.kernel, f1)
                # f1.close()

    def update(self, image, args, frame_id, from_fpga=None, video_name="video1", save_csv=False):
        # image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.normalize:
            image = np.asarray(image[..., np.newaxis]) / 255.0  #gray image normalization
        else:
            image = np.asarray(image[..., np.newaxis])  #gray image normalization

        # search images
        instance_images = [
            self._crop_and_resize(image, self.center, self.x_sz * f, out_size=self.cfg.instance_sz, pad_color=self.avg_color)
            for f in self.scale_factors
        ]
        try:
            instance_images = np.stack(instance_images, axis=0)
        except Exception as e:
            print('error ', e)
        instance_images = torch.from_numpy(instance_images).to(self.device).permute([0, 3, 1, 2]).float()

        # responses
        with torch.set_grad_enabled(False):
            if args.nntype == "ann":
                self.net.eval()
                instances = self.net.feature(instance_images)
            elif args.nntype == "snn":
                self.snn.eval()
                # spike_images_tensor = torch.cat([instance_images for _ in range(args.timesteps)],0)
                spike_images_tensor = GetRateSpikeTensor(instance_images, args.timesteps)
                # spike_images_tensor = GetRateSpikeTensor_new(instance_images,args.timesteps)
                size = spike_images_tensor.shape[2]
                for i in range(3, size, 4):
                    spike_images_tensor[:, :, i, :] = 0
                    spike_images_tensor[:, :, :, i] = 0
                spike_instance_image = SpikeTensor(spike_images_tensor, args.timesteps, scale_factor=1.)
                if save_csv:
                    self.save_csv(spike_instance_image, video=video_name, frame_id=frame_id)
                # instances = self.snn(spike_instance_image)
                out = spike_instance_image
                i = 1
                for layer in self.snn.children():
                    out = layer(out)
                    self.save_out(out, i, frame_id)
                    i += 1
                # f1 = open(f'./snn_debug/pytorch/instance_snn_{frame_id}.data', 'wb')
                # pickle.dump(instances, f1)
                # f1.close()
                if from_fpga is not None:
                    tmp_data = torch.from_numpy(from_fpga).unsqueeze(0).to("cuda:0").float()
                    if tmp_data.sum() != out.data.sum():
                        print(tmp_data.sum())
                        print(out.data.sum())
                        # diff = out.data - tmp_data.data
                        # print(torch.where(diff > 0))
                        print(frame_id)
                    out.data = tmp_data
                instances = out.to_float()

                # f1 = open(f'./snn_debug/pytorch/instance_snn_simple_{frame_id}.data', 'wb')
                # pickle.dump(instances, f1)
                # f1.close()

            else:
                raise ValueError("nn type only support ann/snn.")

        responses = F.conv2d(instances, self.kernel) * 0.001
        # responses = F.conv2d(instances, self.kernel)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack(
            [cv2.resize(t, (self.upscale_sz, self.upscale_sz), interpolation=cv2.INTER_CUBIC) for t in responses], axis=0)
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - self.upscale_sz // 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
            self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2, self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]
        ])

        return box

    def update_simple(self, instances):
        responses = F.conv2d(instances, self.kernel) * 0.001
        # responses = F.conv2d(instances, self.kernel)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack(
            [cv2.resize(t, (self.upscale_sz, self.upscale_sz), interpolation=cv2.INTER_CUBIC) for t in responses], axis=0)
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
                   self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - self.upscale_sz // 2
        disp_in_instance = disp_in_response * \
                           self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
                        self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
                self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2, self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]
        ])

        return box

    def step(self, batch, backward=True, update_lr=False):
        if backward:
            self.net.train()

        else:
            self.net.eval()

        z = batch[0].to(self.device)
        x = batch[1].to(self.device)

        with torch.set_grad_enabled(backward):
            responses = self.net(z, x)
            labels, weights = self._create_labels(responses.size())
            loss = F.binary_cross_entropy_with_logits(responses, labels, weight=weights, reduction='mean')

            if backward:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # if update_lr:

                # cur_lr = self.lr_scheduler.get_last_lr()

        return loss.item()

    def step_snn(self, batch, args, update_lr=False):
        z = batch[0].to(self.device)
        # z = GetRateSpikeTensor_new(z,args.timesteps)
        z = GetRateSpikeTensor(z, args.timesteps)
        z = SpikeTensor(z, args.timesteps, scale_factor=1.)
        x = batch[1].to(self.device)
        # x = GetRateSpikeTensor_new(x,args.timesteps)
        x = GetRateSpikeTensor(x, args.timesteps)
        x = SpikeTensor(x, args.timesteps, scale_factor=1.)

        self.snn.train()
        # if update_lr:
        #     self.lr_scheduler.step()
        z = self.snn(z).to_float()
        x = self.snn(x).to_float()

        n, c, h, w = x.size()
        # fast cross correlation
        x = x.view(1, n * c, h, w)
        out = F.conv2d(x, z, groups=n)
        out = out.view(n, 1, out.size(-2), out.size(-1))
        # adjust the scale of responses
        out = 0.001 * out + 0.0

        labels, weights = self._create_labels(out.size())
        loss = F.binary_cross_entropy_with_logits(out, labels, weight=weights, size_average=True)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = max(round(size), 1)
        corners = np.concatenate((np.round(center - (size - 1) / 2), np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((-corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(image, npad, npad, npad, npad, cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        try:
            # resize to out_size
            patch = cv2.resize(patch, (out_size, out_size))
            if len(patch.shape) < 3:
                patch = patch[..., np.newaxis]
        except:
            patch = patch[..., np.newaxis]

        return patch

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels, self.weights

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos, np.ones_like(x), np.where(dist < r_neg, np.ones_like(x) * 0.5, np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - w // 2
        y = np.arange(h) - h // 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # pos/neg weights
        pos_num = np.sum(labels == 1)
        neg_num = np.sum(labels == 0)
        weights = np.zeros_like(labels)
        weights[labels == 1] = 0.5 / pos_num
        weights[labels == 0] = 0.5 / neg_num
        weights *= pos_num + neg_num

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        weights = weights.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))
        weights = np.tile(weights, [n, c, 1, 1])

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()
        self.weights = torch.from_numpy(weights).to(self.device).float()

        return self.labels, self.weights


if __name__ == '__main__':
    net = SiamFC()
    inputx = torch.randn(1, 1, 255, 255)
    inputz = torch.randn(1, 1, 127, 127)
    sum1 = 0
    sum2 = 0
    sum1 += inputx.shape[0] * inputx.shape[1] * inputx.shape[2] * inputx.shape[3]
    sum2 += inputz.shape[0] * inputz.shape[1] * inputz.shape[2] * inputz.shape[3]
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            inputx = module(inputx)
            inputz = module(inputz)
            sum1 += inputx.shape[0] * inputx.shape[1] * inputx.shape[2] * inputx.shape[3]
            sum2 += inputz.shape[0] * inputz.shape[1] * inputz.shape[2] * inputz.shape[3]

    print('sum1:{}, sum2:{}, all:{}'.format(sum1, sum2, (sum1 + sum2)))
