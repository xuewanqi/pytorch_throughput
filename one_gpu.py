from resnet import resnet50
import argparse
import numpy as np
import os

import torch
import torch.nn as nn

model=resnet50()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--batchsize',type=int, default=16)
    parser.add_argument('--gpu',type=int, default=0)
    args=parser.parse_args()

    steps=args.steps
    batch_size=args.batchsize
    gpu=args.gpu
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device=torch.device('cuda:'+str(gpu))

    model.cuda(device)
    loss_func = nn.CrossEntropyLoss().cuda(device)
    opt=torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,weight_decay=1e-5)

    IMG_SIZE = 224
    x = np.random.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    y = np.random.randint(0, 1000, batch_size, dtype=np.int32)

    tx=torch.tensor(x,dtype=torch.float32).cuda(device)
    ty=torch.tensor(y,dtype=torch.long).cuda(device)

    model.train()

    import time 

    ave_forward_throughput=[]
    ave_backward_throughput=[]

    ave_start=time.time()
    for t in range(steps):
        start=time.time()
        x = model(tx)
        loss = loss_func(x, ty)
        end=time.time()
        fwd_throughput= batch_size/(end-start)
        #print('forward_throughput is {:.4f}'.format(fwd_throughput))
        ave_forward_throughput.append(fwd_throughput)

        start=time.time()
        opt.zero_grad()
        loss.backward()
        opt.step()
        end=time.time()
        bwd_throughput= batch_size/(end-start)
        #print('backward_throughput is {:.4f}'.format(bwd_throughput))
        ave_backward_throughput.append(bwd_throughput)

    ave_end=time.time()
    #print(end-start)
    throughput = steps*batch_size/(ave_end-ave_start)

    ave_fwd_throughput=np.mean(ave_forward_throughput[1:])
    ave_bwd_throughput=np.mean(ave_backward_throughput[1:])

    print('ave_forward_throughput is {:.4f}'.format(ave_fwd_throughput))
    print('ave_backward_throughput is {:.4f}'.format(ave_bwd_throughput))
    print('total_throughput is {:.4f}'.format(throughput))


