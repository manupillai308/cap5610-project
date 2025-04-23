import argparse
import builtins
import math
import os
import random
import string
import shutil
import time
from typing import OrderedDict

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from datetime import datetime

import numpy as np
from dataset.VIGOR import VIGOR_e
from dataset.CVUSA import CVUSA
from dataset.CVACT import CVACT
from model.TransGeo import TransGeo
import h5py

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  #

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH',
                    help='path to save checkpoint (default: none)')
# moco specific configs:
parser.add_argument('--dim', default=1000, type=int,
                    help='feature dimension (default: 128)')

# options for moco v2
parser.add_argument('--cross', action='store_true',
                    help='use cross area')

parser.add_argument('--dataset', default='vigor', type=str,
                    help='vigor, cvusa, cvact')
parser.add_argument('--share', action='store_true',
                    help='share fc')

parser.add_argument('--sat_res', default=0, type=int,
                    help='resolution for satellite')

parser.add_argument('--crop', action='store_true',
                    help='nonuniform crop')

parser.add_argument('--fov', default=0, type=int,
                    help='Fov')

parser.add_argument('--city', default="all", type=str, help="which city to train when doing same area")

best_acc1 = 0


def load_ckpt(ckpt):
    new_ckpt = OrderedDict()
    for key, val in ckpt.items():
        if key.startswith('module.'):
            key = key.replace('module.', '')
        new_ckpt[key] = val
    
    return new_ckpt

def main():
    args = parser.parse_args()
    global best_acc1
    args.gpu = torch.cuda.current_device()
    if not args.cross:
        print(f"Info: training on {args.city} city dataset split/s")

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model
    print("=> creating model '{}'")

    model = TransGeo(args=args)
    model = model.cuda(args.gpu)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            checkpoint['state_dict'] = load_ckpt(checkpoint['state_dict'])
            if not args.crop:
                args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.sat_res != 0:
                pos_embed_reshape = checkpoint['state_dict']['reference_net.pos_embed'][:, 2:, :].reshape(
                    [1,
                     np.sqrt(checkpoint['state_dict']['reference_net.pos_embed'].shape[1] - 2).astype(int),
                     np.sqrt(checkpoint['state_dict']['reference_net.pos_embed'].shape[1] - 2).astype(int),
                     model.reference_net.embed_dim]).permute((0, 3, 1, 2))
                checkpoint['state_dict']['reference_net.pos_embed'] = \
                    torch.cat([checkpoint['state_dict']['reference_net.pos_embed'][:, :2, :],
                               torch.nn.functional.interpolate(pos_embed_reshape, (
                               args.sat_res // model.reference_net.patch_embed.patch_size[0],
                               args.sat_res // model.reference_net.patch_embed.patch_size[1]),
                                                               mode='bilinear').permute((0, 2, 3, 1)).reshape(
                                   [1, -1, model.reference_net.embed_dim])], dim=1)

            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.dataset == 'vigor':
        dataset_cls = VIGOR_e
    else:
        raise ValueError(f'{args.dataset} not found')

    
    # query_loader = torch.utils.data.DataLoader(
    #     dataset_cls(mode='query', args=args), batch_size=args.batch_size,
    #     num_workers=args.workers, pin_memory=True, drop_last=False)
    
    ref_loader = torch.utils.data.DataLoader(
        dataset_cls(mode='ref', args=args), batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    batch_time = AverageMeter('Time', ':6.3f')
    # progress_q = ProgressMeter(
    #     len(query_loader),
    #     [batch_time],
    #     prefix='Test_query: ')
    progress_k = ProgressMeter(
        len(ref_loader),
        [batch_time],
        prefix='Test_reference: ')

    # switch to evaluate mode
    # model_query = model.query_net
    model_reference = model.reference_net
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        # model_query.cuda(args.gpu)
        model_reference.cuda(args.gpu)

    # model_query.eval()
    model_reference.eval()
    print('model validate on cuda', args.gpu)

    with torch.no_grad():
        end = time.time()
        # reference features
        for i, (images, filenames) in enumerate(ref_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            reference_embed = model_reference.evaluate(x=images)  # delta

            for j in range(len(reference_embed)):
                os.makedirs(os.path.join(args.save_path, os.path.join(*(os.path.split(filenames[j])[:-1]))), exist_ok=True)
                to_path = os.path.join(args.save_path, filenames[j] + '.npy')
                np.save(to_path, reference_embed[j].detach().cpu().numpy())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress_k.display(i)

        end = time.time()

        # query features
        # for i, (images, filenames) in enumerate(query_loader):
        #     if args.gpu is not None:
        #         images = images.cuda(args.gpu, non_blocking=True)

        #     # compute output
        #     query_embed = model_query(images)

        #     for j in range(len(query_embed)):
        #         to_path = os.path.join(args.save_path, filenames[j] + 'npy')
        #         np.save(to_path, query_embed[j].detach().cpu().numpy())


        #     # measure elapsed time
        #     batch_time.update(time.time() - end)
        #     end = time.time()

        #     if i % args.print_freq == 0:
        #         progress_q.display(i)

    return None


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
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
