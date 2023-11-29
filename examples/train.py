import argparse
import datetime
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import CIFAR10

from model import pyramidnet

parser = argparse.ArgumentParser(description="cifar10 classification models")
parser.add_argument("--lr", default=0.1, help="")
parser.add_argument("--batch_size", type=int, default=768, help="")
parser.add_argument("--num_workers", type=int, default=4, help="")
parser.add_argument("--gpu_devices", type=int, nargs="+", default=None, help="")

parser.add_argument("--rootdir", type=str, help="")
parser.add_argument("--dist-url", default="tcp://127.0.0.1:3456", type=str, help="")
parser.add_argument("--dist-backend", default="nccl", type=str, help="")
parser.add_argument("--rank", default=0, type=int, help="")
parser.add_argument("--world_size", default=1, type=int, help="")


def main():
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(id) for id in args.gpu_devices])

    ngpus_per_node = torch.cuda.device_count()

    args.world_size = ngpus_per_node * args.world_size
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args,))


def main_worker(gpu, args):
    print("Use GPU: {} for training".format(gpu))

    ngpus_per_node = torch.cuda.device_count()

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank * ngpus_per_node + gpu,
    )

    print("==> Making model..")
    torch.cuda.set_device(gpu)
    net = pyramidnet().to(f"cuda:{gpu}")
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu])

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("The number of parameters of model is", num_params)

    print("==> Preparing data..")
    transforms_train = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    dataset_train = CIFAR10(
        root="../data",
        train=True,
        download=True,
        transform=transforms_train,
    )
    train_sampler = DistributedSampler(dataset_train)
    train_loader = DataLoader(
        dataset_train,
        batch_size=int(args.batch_size / ngpus_per_node),
        shuffle=(train_sampler is None),
        num_workers=int(args.num_workers / ngpus_per_node),
        sampler=train_sampler,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    train(net, criterion, optimizer, train_loader, gpu)


def train(net, criterion, optimizer, train_loader, device):
    net.train()

    train_loss = 0
    correct = 0
    total = 0

    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        start = time.time()

        inputs = inputs.cuda(device)
        targets = targets.cuda(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100 * correct / total

        batch_time = time.time() - start

        if batch_idx % 20 == 0:
            print(
                "Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | batch time: {:.3f}s ".format(
                    batch_idx, len(train_loader), train_loss / (batch_idx + 1), acc, batch_time
                )
            )

    elapse_time = time.time() - epoch_start
    elapse_time = datetime.timedelta(seconds=elapse_time)
    print("Training time {}".format(elapse_time))


if __name__ == "__main__":
    main()
