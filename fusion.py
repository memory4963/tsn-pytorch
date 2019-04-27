import argparse
import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from dataset import TSNDataSet
from models import TSN
from transforms import *
from fusion_opts import parser

best_prec1 = 0


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    global args, best_prec1
    args = parser.parse_args()

    if args.dataset == 'ucf101':
        num_class = 101
    elif args.dataset == 'hmdb51':
        num_class = 51
    elif args.dataset == 'kinetics':
        num_class = 400
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    model = TSN(num_class, args.num_segments, "RGB",
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout,
                partial_bn=not args.no_partialbn)
    model2 = TSN(num_class, args.num_segments, "Flow",
                 base_model=args.arch,
                 consensus_type=args.consensus_type, dropout=args.dropout,
                 partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    train_augmentation = model.get_augmentation()

    model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()
    model2 = torch.nn.DataParallel(model2, device_ids=args.gpus).cuda()

    if args.resume_rgb:
        if os.path.isfile(args.resume_rgb):
            print(("=> loading checkpoint '{}'".format(args.resume_rgb)))
            checkpoint = torch.load(args.resume_rgb)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}'".format(args.resume_rgb)))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume_rgb)))

    if args.resume_flow:
        if os.path.isfile(args.resume_flow):
            print(("=> loading checkpoint '{}'".format(args.resume_flow)))
            checkpoint = torch.load(args.resume_flow)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model2.load_state_dict(checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}'".format(args.resume_flow)))

        else:
            print(("=> no checkpoint found at '{}'".format(args.resume_flow)))

    cudnn.benchmark = True

    # Data loading code
    normalize = GroupNormalize(input_mean, input_std)

    val_loader = torch.utils.data.DataLoader(
        TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=1,
                   modality="RGB",
                   image_tmpl="frame{:06d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader2 = torch.utils.data.DataLoader(
        TSNDataSet("", args.val_list2, num_segments=args.num_segments,
                   new_length=5,
                   modality="Flow",
                   image_tmpl="frame{:06d}.jpg",
                   random_shift=False,
                   transform=torchvision.transforms.Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       normalize,
                   ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        criterion = torch.nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError("Unknown loss type")

    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    validate(val_loader, val_loader2, model, model2, criterion)


def validate(rgb_loader, flow_loader2, model, model2, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model2.eval()

    end = time.time()
    rgb_iter = iter(rgb_loader)
    flow_iter = iter(flow_loader2)
    for j in range(len(rgb_loader)):
        input, target = rgb_iter.next()
        i, t = flow_iter.next()

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        t = t.cuda(async=True)
        i_var = torch.autograd.Variable(i, volatile=True)
        t_var = torch.autograd.Variable(t, volatile=True)

        # compute output
        output = model(input_var)
        o = model2(i_var)
        output += o
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if j % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                j, len(rgb_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
           .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
