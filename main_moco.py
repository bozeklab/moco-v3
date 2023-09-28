#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import torchvision.models as torchvision_models
from torch.distributed.run import get_args_parser
from torch.utils.tensorboard import SummaryWriter
import PIL
import numpy as np
import os
import time
from pathlib import Path

import moco.builder
import moco.loader
import moco.optimizer

import vits
from util import misc
from util.augmentation import DataAugmentationForSIMTraining
from util.datasets import ImgWithPickledBoxesDataset
import numpy as np
import argparse
import datetime
import json
import random

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names


def get_args_parser():
    parser = argparse.ArgumentParser(description='MoCo ImageNet Pre-Training')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
    parser.add_argument('--num_boxes', default=150, type=int,
                        help='maximal number of boxes')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--data_path', default='/data/pwojcik/images_he_seg/positive', type=str,
                        help='dataset path')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', default=4096, type=int,
                        metavar='N',
                        help='mini-batch size (default: 4096), this is the total '
                             'batch size of all GPUs on all nodes when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.6, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                        metavar='W', help='weight decay (default: 1e-6)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=256, type=int,
                        help='feature dimension (default: 256)')
    parser.add_argument('--moco-mlp-dim', default=4096, type=int,
                        help='hidden dimension in MLPs (default: 4096)')
    parser.add_argument('--moco-m', default=0.99, type=float,
                        help='moco momentum of updating momentum encoder (default: 0.99)')
    parser.add_argument('--moco-m-cos', action='store_true',
                        help='gradually increase moco momentum to 1 with a '
                             'half-cycle cosine schedule')

    parser.add_argument('--moco-t', default=1.0, type=float,
                        help='softmax temperature (default: 1.0)')

    # vit specific configs:
    parser.add_argument('--stop-grad-conv1', action='store_true',
                        help='stop-grad after first conv, or patch embedding')

    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')

    # other upgrades
    parser.add_argument('--optimizer', default='lars', type=str,
                        choices=['lars', 'adamw'],
                        help='optimizer used (default: lars)')
    parser.add_argument('--warmup-epochs', default=10, type=int, metavar='N',
                        help='number of warmup epochs')
    parser.add_argument('--crop-min', default=0.08, type=float,
                        help='minimum scale for random cropping (default: 0.08)')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--auto_resume', default=True)
    parser.add_argument('--save_freq', default=50, type=int)
    parser.add_argument('--save_latest_freq', default=1, type=int)
    parser.add_argument('--fp32', default=False, action='store_true')
    parser.add_argument('--amp_growth_interval', default=2000, type=int)

    return parser


def main(args):
    misc.init_distributed_mode(args)  # need change to torch.engine

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # disable tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    transform_train = DataAugmentationForSIMTraining(args)
    print(f'Pre-train data transform:\n{transform_train}')

    train_dataset = ImgWithPickledBoxesDataset(os.path.join(args.data_path), transform=transform_train)
    print(f'Build dataset: train images = {len(train_dataset)}')

    # build dataloader
    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        model = moco.builder.MoCo_ViT(
            partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)
    else:
        model = moco.builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True),
            args.moco_dim, args.moco_mlp_dim, args.moco_t)

    model.to(device)
    model_without_ddp = model
    args.lr = args.lr * args.batch_size / 256

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    print("Model = %s" % str(model_without_ddp))
    if args.optimizer == 'lars':
        optimizer = moco.optimizer.LARS(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler(enabled=(not args.fp32), growth_interval=args.amp_growth_interval)

    misc.auto_load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer,
                         loss_scaler=loss_scaler)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # train for one epoch
        train(data_loader_train, model, optimizer, loss_scaler, log_writer, epoch, args)

        dist.barrier()

        # save ckpt
        if args.output_dir and ((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
        if (epoch + 1) % args.save_latest_freq == 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, latest=True)

        # log information
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        if misc.is_main_process():
            epoch_total_time = time.time() - epoch_start_time
            now = datetime.datetime.today()
            eta = now + datetime.timedelta(seconds=(args.epochs - epoch - 1) * int(epoch_total_time))
            next_50_ep = ((epoch + 1) // 50 + 1) * 50
            eta_to_next_50 = now + datetime.timedelta(seconds=(next_50_ep - epoch - 1) * int(epoch_total_time))
            print(f"ETA to {args.epochs:4d}ep:\t{str(eta)}")
            print(f"ETA to {next_50_ep:4d}ep:\t{str(eta_to_next_50)}")
        dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train(train_loader, model, optimizer, scaler, log_writer, epoch, args):
    # switch to train mode
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m
    for data_iter_step, sample in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        # measure data loading time

        # adjust learning rate and momentum coefficient per iteration
        lr = adjust_learning_rate(optimizer, epoch + data_iter_step / iters_per_epoch, args)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + data_iter_step / iters_per_epoch, args)

        x1 = sample['x1']
        x2 = sample['x2']
        boxes1 = sample['boxes1']
        boxes2 = sample['boxes2']
        if args.gpu is not None:
            x1 = x1.cuda(args.gpu, non_blocking=True)
            x2 = x2.cuda(args.gpu, non_blocking=True)
            boxes1 = boxes1.cuda(args.gpu, non_blocking=True)
            boxes2 = boxes2.cuda(args.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            loss = model(x1, x2, boxes1, boxes2, moco_m)

        loss_value = loss.item()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        print('dupa')

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        outputs_reduced = {k_: misc.all_reduce_mean(v_) for k_, v_ in outputs.items()}
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(train_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            if scaler is not None:
                log_writer.add_scalar('loss_scale', scaler, epoch_1000x)
            for k_, v_ in outputs_reduced.items():
                log_writer.add_scalar(f'train/{k_}', v_, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """Adjust moco momentum based on current epoch"""
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

