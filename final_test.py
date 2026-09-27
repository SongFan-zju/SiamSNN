# !usr/bin/env python3
# -*- coding:utf-8 -*-

#############################################
# FilePath: /SiamSNN/snn_demo.py
# Date: 2021-12-14
# Author: FanJH
# Description:
#############################################
from __future__ import absolute_import
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import glob
import copy
import json
import cv2
import argparse
import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchsummary import summary

from got10k.datasets import ImageNetVID, GOT10k
from pairwise import Pairwise
from siamfc import TrackerSiamFC
from spiking_utils.snn_transformer import SNNTransformer


class ListWrapper(nn.Module):
    """
    partial model
    """

    def __init__(self, modulelist):
        super().__init__()
        self.list = modulelist

    def forward(self, x):
        for layer in self.list:
            x = layer(x)

        return x


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, w1_1, h1_1) = bbox1
    (x0_2, y0_2, w1_2, h1_2) = bbox2
    x1_1 = x0_1 + w1_1
    x1_2 = x0_2 + w1_2
    y1_1 = y0_1 + h1_1
    y1_2 = y0_2 + h1_2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def not_exist(pred):
    return (len(pred) == 1 and pred[0] == 0) or len(pred) == 0


def eval(out_res, label_res):
    measure_per_frame = []
    cnt = 0
    for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
        measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt) if len(_pred) > 1 else 0)
        cnt = cnt + 1 if measure_per_frame[-1] > 0.4 else cnt
    return np.mean(measure_per_frame), cnt / len(measure_per_frame)


def main(args):
    # setup dataset
    name = args.dataset
    normalize = False
    epoch_num = 10

    exemplar_size = 127  #64  # 127
    instance_size = 255  #128  # 255
    response_sz = 9  #9 #17

    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)

    # setup tracker net_path=motdel.pth exemplar_sz=127 instance_sz=255
    tracker = TrackerSiamFC(normalize,
                            epoch_num,
                            net_path=args.checkpoint,
                            exemplar_sz=exemplar_size,
                            instance_sz=instance_size,
                            response_sz=response_sz,
                            scale_num=1)
    # build snn
    if args.nntype == "snn":
        tracker.snn = torch.load('checkpoint_snn_input255_exp2_1\model_zz_snn_e100.pth')

    #
    tracker.net.to(device)
    # summary(tracker.net.feature, (1, args.instance_size, args.instance_size))
    # summary(tracker.net.feature, (1, args.exemplar_size, args.exemplar_size))
    video_paths = [
        os.path.join("videos", "video1"),
        os.path.join("videos", "video2"),
        os.path.join("videos", "video3"),
        os.path.join("videos", "video4"),
    ]
    save_csv = False
    video_num = len(video_paths)
    output_dir = os.path.join('results', tracker.name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    overall_performance = []
    overall_map50 = []
    # run tracking experiments and report performance
    input_type = args.input_type
    for video_id, video_path in enumerate(video_paths, start=1):
        video_name = os.path.basename(video_path)
        # video_file = os.path.join(video_path, '%s.mp4' % args.video_mode)
        # if input_type == 'videos':
        video_file = os.path.join(video_path, '%s.mp4' % args.video_mode)
        # else:
        img_list = os.listdir(os.sep.join([video_path, 'imgs']))
        img_list.sort()

        res_file = os.path.join(video_path, '%s_label.json' % args.video_mode)
        with open(res_file, 'r') as f:
            label_res = json.load(f)

        init_rect = label_res['gt_rect'][0]
        # if input_type == 'videos':
        capture = cv2.VideoCapture(video_file)
        if args.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = capture.get(cv2.CAP_PROP_FPS)
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            video_writer = cv2.VideoWriter("results/Video/" + video_name + "_" + args.nntype + "res_debug.mp4", fourcc, fps,
                                           size)
        frame_id = 0
        out_res = []
        exit_flag = False
        if input_type == 'imgs':
            imgs_num = len(img_list)
        while True:
            if input_type == 'videos':
                ret, frame = capture.read()
                if not ret:
                    break
            else:
                if frame_id >= imgs_num:
                    break
                frame = cv2.imread(os.sep.join([video_path, 'imgs', img_list[frame_id]]))
                if frame is None:
                    break
            # print('frame id:', frame_id)

            if frame_id == 0:
                tracker.init(frame, init_rect, args, frame_id, video_name=video_path.split("\\")[-1], save_csv=save_csv)
                out = init_rect
                out_res.append(init_rect)

                f1 = open(f'./snn_debug/fpga/exemplar_snn_simple_{frame_id}.data', 'rb')
                tracker.kernel = pickle.load(f1)
                f1.close()

                if args.save_snn_weights:
                    tracker.snn.save_conv_weights("checkpoint/snn0/", "darwin")
            elif frame_id % 1 == 0:
                # frame_inv = 255 - frame
                # if frame_id == 214:
                #     print('debug')
                out = tracker.update(frame, args, frame_id, video_name=video_path.split("\\")[-1], save_csv=save_csv)
                out_res.append(out.tolist())
            # if args.visulization:
            _gt = label_res['gt_rect'][frame_id]
            # _gt = out_res[frame_id]
            _exist = label_res['exist'][frame_id]
            if _exist:
                cv2.rectangle(frame, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])), (0, 255, 0),
                              thickness=2)
            cv2.putText(frame, 'exist' if _exist else 'not exist', (frame.shape[1] // 2 - 20, 30), 1, 2,
                        (0, 255, 0) if _exist else (0, 0, 255), 2)

            cv2.rectangle(frame, (int(out[0]) - 5, int(out[1] - 5)), (int(out[0] + out[2] + 5), int(out[1] + out[3] + 5)),
                          (255, 0, 0),
                          thickness=2)
            if args.save_video:
                video_writer.write(frame)

                # res_img_dir = 'results/imgs/'+video_name
                # os.makedirs(res_img_dir, exist_ok=True)
                # img_name = "{:0>4d}.png".format(frame_id)
                # cv2.imwrite(os.sep.join([res_img_dir, img_name]), frame)
            if args.visulization:
                cv2.imshow(video_name, frame)
                c = cv2.waitKey(1) & 0xff
                if c == ord(' '):
                    # capture.release()
                    break
                elif c == 27:
                    exit_flag = True
                    # capture.release()
                    break

            frame_id += 1
        capture.release()
        if args.visulization:
            cv2.destroyAllWindows()
        if args.save_video:
            video_writer.release()
        # save result
        output_file = os.path.join(output_dir, '%s_%s.txt' % (video_name, args.video_mode))
        with open(output_file, 'w') as f:
            json.dump({'res': out_res}, f)

        mixed_measure, cnt = eval(out_res, label_res)
        overall_performance.append(mixed_measure)
        overall_map50.append(cnt)
        # print('[%03d/%03d] %20s %5s IoU: %.03f' % (video_id, video_num, video_name, args.video_mode, mixed_measure))
        print('[%03d/%03d] %20s 跟踪准确率: %.03f' % (video_id, video_num, video_name, cnt))
        #interrupt video track
        if exit_flag:
            break

    # print('[Overall] %5s IoU: %.03f' % (args.video_mode, np.mean(overall_performance)))
    print('[Overall]  平均跟踪准确率 Avg_ACC: %.03f\n' % (np.mean(overall_map50)))

    if args.save_snn_weights:
        torch.save(tracker.snn, "checkpoint/model_snn.pt")
        tracker.snn.save_conv_weights("checkpoint/snn/", "darwin")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nntype', type=str, default="snn", choices=["ann", "snn"], help="use ann or snn inference.")
    parser.add_argument('--dataset', type=str, default="GOT-10k", choices=["GOT-10k", "VID"], help="dataset type.")
    parser.add_argument('--batch_size', type=int, default=32, help="data batch size.")
    parser.add_argument('--instance_size', type=int, default=255, choices=[255, 128], help="search area size.")
    parser.add_argument('--exemplar_size', type=int, default=127, choices=[127, 64], help="the exemplar image size.")
    parser.add_argument('--response_sz', type=int, default=17, choices=[17, 9], help="the response size.")
    parser.add_argument('--checkpoint', type=str, default="checkpoint\model_e50.pth", help="weights file path")

    parser.add_argument('--timesteps', type=int, default=12, help="spiking time window size")
    parser.add_argument('--reset_mode',
                        type=str,
                        default="subtraction",
                        choices=['zero', 'subtraction'],
                        help="potential reset mode")
    parser.add_argument('--channel_wise', '-cw', type=bool, default=True, help="transform in each channel")
    parser.add_argument('--statistics_iters', type=int, default=8, help="iterations for gather activation statistics")
    parser.add_argument('--video_mode', type=str, default="IR", choices=["IR", "RGB"], help="video sample mode.")
    parser.add_argument('--visulization', type=bool, default=True, help="show video.")
    parser.add_argument('--save_video', type=bool, default=True, help="save the result video.")
    parser.add_argument('--device', type=str, default='0', help="device id (i.e. 0 or 1 or cpu)")
    parser.add_argument('--save_snn_weights', type=bool, default=False, help="save the result video.")
    parser.add_argument('--input_type',
                        type=str,
                        default='imgs',
                        choices=['imgs', 'videos'],
                        help='select input type:video or img sequence')

    args = parser.parse_args()

    main(args)
