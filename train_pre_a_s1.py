from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from awb import datasets
from awb import models_pre_a_s1
from awb.trainers import MMTTrainer
from awb.evaluators import Evaluator, extract_features
from awb.utils.data import IterLoader
from awb.utils.data import transforms as T
from awb.utils.data.sampler import RandomMultipleGallerySampler
from awb.utils.data.preprocessor import Preprocessor
from awb.utils.logging import Logger
from awb.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict


start_epoch = best_mAP = 0

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset

def get_train_loader(dataset, height, width, batch_size, workers,
                    num_instances, iters):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = dataset.train
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer, mutual=True),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def create_model(args):
    model_1 = models_pre_a_s1.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)
    model_2 = models_pre_a_s1.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)

    model_1_ema = models_pre_a_s1.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)
    model_2_ema = models_pre_a_s1.create(args.arch, num_features=args.features, dropout=args.dropout, num_classes=args.num_clusters)

    model_1.cuda()
    model_2.cuda()
    model_1_ema.cuda()
    model_2_ema.cuda()
    model_1 = nn.DataParallel(model_1)
    model_2 = nn.DataParallel(model_2)
    model_1_ema = nn.DataParallel(model_1_ema)
    model_2_ema = nn.DataParallel(model_2_ema)
    
    initial_weights = load_checkpoint(args.init_1)
    copy_state_dict(initial_weights['state_dict'], model_1)
    copy_state_dict(initial_weights['state_dict'], model_1_ema)
    model_1_ema.module.classifier.weight.data.copy_(model_1.module.classifier.weight.data)

    initial_weights = load_checkpoint(args.init_2)
    copy_state_dict(initial_weights['state_dict'], model_2)
    copy_state_dict(initial_weights['state_dict'], model_2_ema)
    model_2_ema.module.classifier.weight.data.copy_(model_2.module.classifier.weight.data)
    
    
    for param in model_1_ema.parameters():
        param.detach_()
    for param in model_2_ema.parameters():
        param.detach_()

    return model_1, model_2, model_1_ema, model_2_ema


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    dataset_target = get_data(args.dataset_target, args.data_dir)
    test_loader_target = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model_1, model_2, model_1_ema, model_2_ema = create_model(args)

    # Evaluator
    evaluator_1_ema = Evaluator(model_1_ema)
    evaluator_2_ema = Evaluator(model_2_ema)

    clusters = [args.num_clusters]*args.epochs
    feature_length = args.features if args.features>0 else 2048
    moving_avg_features = np.zeros((len(dataset_target.train), feature_length))

    for nc in range(len(clusters)):
        cluster_loader = get_test_loader(dataset_target, args.height, args.width, args.batch_size, args.workers, testset=dataset_target.train)

        dict_f, _ = extract_features(model_1_ema, cluster_loader, print_freq=50)
        cf_1 = torch.stack(list(dict_f.values())).numpy()
        dict_f, _ = extract_features(model_2_ema, cluster_loader, print_freq=50)
        cf_2 = torch.stack(list(dict_f.values())).numpy()
        cf = (cf_1+cf_2)/2
        moving_avg_features = moving_avg_features.astype(np.float32)
        cf = cf.astype(np.float32)
        moving_avg_features = moving_avg_features*args.moving_avg_momentum+cf*(1-args.moving_avg_momentum)
        moving_avg_features = moving_avg_features / (1-args.moving_avg_momentum**(nc+1))

        print('\n Clustering into {} classes \n'.format(clusters[nc]))
        km = KMeans(n_clusters=clusters[nc], random_state=args.seed, n_jobs=2).fit(moving_avg_features)

        model_1.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        model_2.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        model_1_ema.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())
        model_2_ema.module.classifier.weight.data.copy_(torch.from_numpy(normalize(km.cluster_centers_, axis=1)).float().cuda())

        target_label = km.labels_

        # change pseudo labels
        for i in range(len(dataset_target.train)):
            dataset_target.train[i] = list(dataset_target.train[i])
            dataset_target.train[i][1] = int(target_label[i])
            dataset_target.train[i] = tuple(dataset_target.train[i])

        train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters)

        # Optimizer
        params = []
        for key, value in model_1.named_parameters():
            if not value.requires_grad:
                continue
            if 'cbam' in key:
                params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        for key, value in model_2.named_parameters():
            if not value.requires_grad:
                continue
            if 'cbam' in key:
                params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
        optimizer = torch.optim.Adam(params)

        # Trainer
        trainer = MMTTrainer(model_1, model_2, model_1_ema, model_2_ema,
                                num_cluster=clusters[nc], alpha=args.alpha)

        train_loader_target.new_epoch()
        epoch = nc

        trainer.train(epoch, train_loader_target, optimizer,
                    ce_soft_weight=args.soft_ce_weight, tri_soft_weight=args.soft_tri_weight,
                    print_freq=args.print_freq, train_iters=len(train_loader_target))

        def save_model(model_ema, is_best, best_mAP, mid):
            save_checkpoint({
                'state_dict': model_ema.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model'+str(mid)+'_checkpoint.pth.tar'))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            mAP_1 = evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
            mAP_2 = evaluator_2_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=False)
            is_best = (mAP_1>best_mAP) or (mAP_2>best_mAP)
            best_mAP = max(mAP_1, mAP_2, best_mAP)
            save_model(model_1_ema, (is_best and (mAP_1>mAP_2)), best_mAP, 1)
            save_model(model_2_ema, (is_best and (mAP_1<=mAP_2)), best_mAP, 2)

            print('\n * Finished epoch {:3d}  model no.1 mAP: {:5.1%} model no.2 mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP_1, mAP_2, best_mAP, ' *' if is_best else ''))

    print ('Test on the best model.')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model_1_ema.load_state_dict(checkpoint['state_dict'])
    evaluator_1_ema.evaluate(test_loader_target, dataset_target.query, dataset_target.gallery, cmc_flag=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMT Training")
    # data
    parser.add_argument('-dt', '--dataset-target', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--num-clusters', type=int, default=500)
    parser.add_argument('--height', type=int, default=256,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models_pre_a_s1.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--moving-avg-momentum', type=float, default=0)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--soft-ce-weight', type=float, default=0.5)
    parser.add_argument('--soft-tri-weight', type=float, default=0.8)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=800)
    # training configs
    parser.add_argument('--init-1', type=str, default='', metavar='PATH')
    parser.add_argument('--init-2', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    parser.add_argument('--eval-step', type=int, default=1)
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
