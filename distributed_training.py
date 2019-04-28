import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models

from torch.utils.data import Dataset, DataLoader

from resnet import resnet50


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--steps', default=500, type=int)


best_acc1 = 0


def main():
    args = parser.parse_args()

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed #True

    ngpus_per_node = torch.cuda.device_count() #8

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size  #n*8
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    # else:
    #     # Simply call main_worker function
    #     main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    #global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        # if args.dist_url == "env://" and args.rank == -1:
        #     args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            torch.cuda.set_device(args.gpu)
           
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
 
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()

    model=resnet50()
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            #args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
           
        # else:
        #     model.cuda()
        #     # DistributedDataParallel will divide and allocate batch_size to all
        #     # available GPUs if device_ids are not set
        #     model = torch.nn.parallel.DistributedDataParallel(model)
    # elif args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model = model.cuda(args.gpu)
    # else:
    #     # DataParallel will divide and allocate batch_size to all available GPUs
    #     if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #         model.features = torch.nn.DataParallel(model.features)
    #         model.cuda()
    #     else:
    #         model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    print('finish initial model')
    # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc1 = checkpoint['best_acc1']
    #         if args.gpu is not None:
    #             # best_acc1 may be from a checkpoint from a different GPU
    #             best_acc1 = best_acc1.to(args.gpu)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    #cudnn.benchmark = True

    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # class RandomDataset(Dataset):

    # def __init__(self, length):
    #     self.len = length
    #     #self.data = torch.randn(length, 3,224,224)
    #     IMG_SIZE = 224
    #     x = np.random.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    #     y = np.random.randint(0, 1000, batch_size, dtype=np.int32)

    #     tx=torch.tensor(x,dtype=torch.float32).to(device)
    #     ty=torch.tensor(y,dtype=torch.long).to(device)

    # def __getitem__(self, index):
    #     return self.data

    # def __len__(self):
    #     return self.len


    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # # else:
    # #     train_sampler = None

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
    #     num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # val_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(valdir, transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=True)

    # if args.evaluate:
    #     validate(val_loader, model, criterion, args)
    #     return
    import numpy as np
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        #adjust_learning_rate(optimizer, epoch, args)

        IMG_SIZE = 224
        x = np.random.randn(args.batch_size, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
        y = np.random.randint(0, 1000, args.batch_size, dtype=np.int32)

        input=torch.tensor(x,dtype=torch.float32)
        target=torch.tensor(y,dtype=torch.long)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        inputs=(input,target)

        # train for one epoch
        model.train()
        ave_start, ave_forward_throughput,ave_backward_throughput=train(inputs,args.steps, model, criterion, optimizer, epoch, args)

    ave_end=time.time()
    throughput = (args.steps-5)*args.batch_size*args.world_size/(ave_end-ave_start)

    ave_fwd_throughput=np.mean(ave_forward_throughput[5:])
    ave_bwd_throughput=np.mean(ave_backward_throughput[5:])

    print('ave_forward_throughput is {:.4f}'.format(ave_fwd_throughput))
    print('ave_backward_throughput is {:.4f}'.format(ave_bwd_throughput))
    print('total_throughput is {:.4f}'.format(throughput))

        # evaluate on validation set
        # acc1 = validate(val_loader, model, criterion, args)

        # # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)

        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer' : optimizer.state_dict(),
        #     }, is_best)


def train(inputs, steps, model, criterion, optimizer, epoch, args):
    # batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
    #                          top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    #model.train()

    #end = time.time()
    #for i, (input, target) in enumerate(train_loader):
    ave_forward_throughput=[]
    ave_backward_throughput=[]
    for i in range(steps):
        # measure data loading time
        #data_time.update(time.time() - end)

        # if args.gpu is not None:
        #     input = input.cuda(args.gpu, non_blocking=True)
        # target = target.cuda(args.gpu, non_blocking=True)

        if i==4:
            ave_start=time.time()
        start=time.time()

        output = model(inputs[0])
        loss = criterion(output, inputs[1])
        
        end=time.time()
        fwd_throughput= args.batch_size*args.world_size/(end-start)
        #print('forward_throughput is {:.4f}'.format(fwd_throughput))
        ave_forward_throughput.append(fwd_throughput)

        start=time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end=time.time()
        bwd_throughput= args.batch_size*args.world_size/(end-start)
        #print('backward_throughput is {:.4f}'.format(bwd_throughput))
        ave_backward_throughput.append(bwd_throughput)

    return ave_start, ave_forward_throughput,ave_backward_throughput

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    main()
