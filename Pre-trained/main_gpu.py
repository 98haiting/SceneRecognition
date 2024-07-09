import argparse
import os
import warnings
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import pandas as pd
import random
import numpy as np
from collections import OrderedDict, Counter


from model import models
from data import ChallengeDataset, TestDataset
from trainer_ngpus import Trainer
from tester import Tester
from sklearn.model_selection import train_test_split, KFold, cross_val_score

def parse_args():
    """
    This function is to define some arguments for the model
    :return: properties about training the model
    """
    parser = argparse.ArgumentParser(description='Pytorch Places365 Retraining on Pre-trained model')
    parser.add_argument('--model', type=str, default='vit_h14', help='model to retrain (default: vit_h14_in1k)')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='Batch size',
                        help='total batch size of all GPUs on current node when using Data Parallel or Distributed Data Parallel (default: 256)')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency (default: 10)')
    # world size: number of gpus/global processes: 3
    parser.add_argument('--world-size', type=int, default=-1, help='number of nodes for distributed training')
    # rank: identifier of the current gpu/identifier of processor: 1, 2, 3
    parser.add_argument('--rank', type=int, default=-1, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
    # ncc: communication between gpus
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
    # gpu: identifier of the current gpu
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
    # if accepted, set the parameter to true
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training')

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.003)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=123, help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    # checkpoint path: "./previous_checkpoints/checkpoints/....pkl"
    parser.add_argument('--checkpoints', action='store_true', default=False, help='whether to load the checkpoint')
    parser.add_argument('--stylized', action='store_true', default=False, help='train with stylized images')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='path to the checkpoint')
    parser.add_argument('--pretrained', action='store_true', default=False, help='use of the pre-trained model provided by the authors')
    parser.add_argument('--prediction', action='store_true', default=False, help='prediction mode')
    parser.add_argument('--train_art', action='store_true', default=False, help='whether to train with Artplaces dataset')
    parser.add_argument('--weighting', action='store_true', default=False, help='loss weighting')
    parser.add_argument('--result_config', type=str, default=None, help='quantitative results and confusion matrix')

    return parser.parse_args()

def main():
    args = parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ['RANK'])
    args.local_rank = int(os.environ['LOCAL_RANK'])

    print("in the main function")
    print(f"world size: {args.world_size}, rank: {args.rank}, local rank: {args.local_rank}")

    ngpus_per_node = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main_worker(args.local_rank, args)

def main_worker(gpu, args):
    args.gpu = gpu

    dist.init_process_group(backend=args.dist_backend)
    print("process group initialized")

    # loading model and processing to gpu/cpu
    get_model = models()
    model, resolution = get_model.Get_model(model_name=args.model, pretrained=args.pretrained)
    if args.checkpoints:
        args.checkpoint_path = f'/home/woody/iwi5/iwi5154h/FraClf/outputs/{args.checkpoint_path}/checkpoints/'
        state_dict = torch.load(args.checkpoint_path + f'{args.model}_weights.pkl', map_location=torch.device("cuda", args.local_rank))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print("get the model with pre-trained weights")

    if not torch.cuda.is_available():
        print("GPU is not available, using CPU")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    if not args.train_art:
        # Data loading code and preprocessing
        traindir = os.path.join('./data', 'places365_train_standard.txt')
        valdir = os.path.join('./data', 'places365_val.txt')

        with open(traindir, 'r') as f:
            challengeDataset_train = f.readlines()
        with open(valdir, 'r') as f:
            challengeDataset_val = f.readlines()

        # training samples
        train_samples = ChallengeDataset(challengeDataset_train, mode='train', resolution=resolution, args=args)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_samples, shuffle=True)
        train_dataloader = torch.utils.data.DataLoader(train_samples, batch_size=args.batch_size,
                                                    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

        # validation samples
        val_samples = ChallengeDataset(challengeDataset_val, mode='val', resolution=resolution, args=args)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_samples)
        val_dataloader = torch.utils.data.DataLoader(val_samples, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=args.workers, pin_memory=True, sampler=val_sampler)

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)

    else:
        dataset = './Artplace_train.csv'
        challengeDataset_train_val = pd.read_csv(dataset)
        challengeDataset_train, challengeDateset_val = train_test_split(challengeDataset_train_val, test_size=0.2)

        train_samples = ChallengeDataset(challengeDataset_train, mode='train', resolution=resolution, args=args)
        train_dataloader = torch.utils.data.DataLoader(train_samples, batch_size=args.batch_size,
                                                       num_workers=args.workers, pin_memory=True, shuffle=True)

        val_samples = ChallengeDataset(challengeDateset_val, mode='val', resolution=resolution, args=args)
        val_dataloader = torch.utils.data.DataLoader(val_samples, batch_size=args.batch_size,
                                                     shuffle=False, num_workers=args.workers, pin_memory=True)

        # loss weighting configuration
        sum_data = Counter(train_samples)
        nSamples = []
        for label in range(768):
            if label not in sum_data:
                nSamples.append(1e-10)
            else:
                nSamples.append(sum_data[label])

        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]

        for index in range(len(normedWeights)):
            if normedWeights[index] >= 1:
                normedWeights[index] = 1

        normedWeights = torch.FloatTensor(normedWeights).to(device)
        if args.weighting:
            weights = normedWeights
        else:
            weights = None
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum)


    dist.barrier()
    # running configuration
    if not args.prediction:
        if not args.k:
            print("training configuration")
            trainer = Trainer(model=model, train_data=train_dataloader, val_data=val_dataloader, criterion=criterion,
                              optimizer=optimizer, early_stop=5, args=args, device=device)
            print("start training")
            tok1, tok5, loss = trainer.fit(epochs=args.epochs)
        else:
            kf = KFold(n_splits=10)
            score = cross_val_score(model, train_samples, cv=kf)
            print(f"k-fold validation score: {score}")

    # testdir = './image-level-labels - labels.csv'
    testdir_ODOR = os.path.join('./data', 'Artwork1.csv')
    testdir_Art = './Artplace_test_corrected.csv'

    # challengeDataset_test = pd.read_csv(testdir)  # DataFrame with all the info
    challengeDataset_test_ODOR = pd.read_csv(testdir_ODOR)
    challengeDataset_test_Art = pd.read_csv(testdir_Art)
    challengeDataset_test = pd.concat([challengeDataset_test_ODOR, challengeDataset_test_Art], ignore_index=True)

    # test samples
    test_dataloader = torch.utils.data.DataLoader(
        TestDataset(challengeDataset_test, resolution=resolution),
        shuffle=False, num_workers=args.workers, pin_memory=True)

    # prediction
    if args.prediction:
        if args.local_rank == 0:
            print("testing configuration")
            tester = Tester(model=model, test_data=test_dataloader, criterion=criterion, args=args)
            print("start testing")
            top1_test, top5_test = tester.predictor()

if __name__ == '__main__':
    main()


