from __future__ import print_function, division
import torch
import os
import torch.nn as nn
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import sys
import warnings
import time
import pickle

warnings.filterwarnings("ignore")
from model import SG2HOI
import random
import datetime
import torch.distributed as dist
from apex import amp
import pandas as pd
from tqdm import tqdm
from util.checkpoint import DetectronCheckpointer
from solver import make_lr_scheduler
from solver import make_optimizer
from util.model_serialization import load_state_dict
from util.metric_logger import MetricLogger
from util.checkpoint import clip_grad_norm
from solver.trainer import reduce_loss_dict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataloader_vcoco import Rescale, VcocoDataset, vcoco_collate
from utils import mkdir, debug_print, get_side_infos, setup_logger, convert2vcoco, save_checkpoint, \
    calculate_averate_precision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number_of_epochs', type=int, required=False, default=100)
    parser.add_argument('--gpu_id', type=str, required=False, default="0", )
    parser.add_argument('--learning_rate', type=float, required=False, default=0.01, help='Initial learning_rate')
    parser.add_argument('--saving_epoch', '--saving_epoch', type=int, required=False, default=10)
    parser.add_argument('--output', type=str, required=False, default='/mnt/hdd2/hoi_chechpoint/')
    parser.add_argument('--batch_size', type=int, required=False, default=4, help='Batch size')
    parser.add_argument('--resume_model', type=bool, required=False, default=True)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('-num_work', '--num_work', type=int, default=5, required=False,
                        help="number of threads for data_loader.")
    parser.add_argument('-mean_best', '--mean_best', type=float, default=0., required=False)
    parser.add_argument('-start_epoch', '--start_epoch', type=int, default=0, required=False)
    parser.add_argument('-num_epochs', '--num_epochs', type=int, default=100, required=False)
    parser.add_argument('-device', '--device', type=str, default="cuda", required=False)
    parser.add_argument('--data_dir', type=str, default="/mnt/hdd2/hoi_data/All_data_vcoco/", required=False)
    parser.add_argument('--prior', type=str, default="/mnt/hdd2/hoi_data/infos/prior.pickle", required=False)
    parser.add_argument('--train_detected_results', type=str,
                        default="/mnt/hdd2/Object_Detections_vcoco/train_detected_results.pk",
                        required=False)
    parser.add_argument('--val_detected_results', type=str,
                        default="/mnt/hdd2/bject_Detections_vcoco/train_detected_results.pk",
                        required=False)
    parser.add_argument('--test_detected_results', type=str,
                        default="/mnt/hdd2/Object_Detections_vcoco/test_detected_results.pk",
                        required=False)
    parser.add_argument('--sg_data', type=str,
                        default="/mnt/hdd2/Object_Detections_vcoco/vcoco_", required=False,help="Your scene graph predicted results' path.")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    data_loaders = construct_dataloaders(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda")
    model = SG2HOI(args).to(device)
    logger = setup_logger("SG2HOI", args.output, 0)
    trainables = []
    for name, p in model.named_parameters():
        if name.split('.')[0] in ['Conv_pretrain']:
            p.requires_grad = False
        else:
            trainables.append(p)
    if args.output:
        mkdir(args.output)
    optimizer = optim.SGD([{"params": trainables, "lr": args.learning_rate}], momentum=0.9, weight_decay=0.0001)
    lambd = lambda epoch: 1.0 if epoch < 10 else (10 if epoch < 33 else 1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, [lambd])
    debug_print(logger, 'end optimizer and shcedule')
    amp_opt_level = 'O0'  # "O1" for mixture precision training.
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)
    meters = MetricLogger(delimiter="  ")

    if model.use_faster_rcnn_backbone:
        load_mapping = {"box_feature_extractor": "roi_heads.box.feature_extractor", 'backbone_net': 'backbone'}
        checkpoint = torch.load('/mnt/hdd2/datasets/vg/pretrained_faster_rcnn/model_final.pth',
                                map_location=torch.device("cpu"))
        load_state_dict(model, checkpoint.pop("model"), load_mapping)

    if args.resume_model is True:
        try:
            checkpoint = torch.load(args.output + '/bestcheckpoint.pth.tar')
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args.start_epoch = checkpoint['epoch']
            args.mean_best = checkpoint['mean_best']
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(
                "=> loaded checkpoint when best_prediction {} and epoch {}".format(args.mean_best, checkpoint['epoch']))
        except:
            print('Failed to load checkPoint')
            raise

    train_test(model, optimizer, scheduler, data_loaders, logger, meters, args)


def construct_dataloaders(cfg):
    annotation_train = cfg.data_dir + 'Annotations_vcoco/train_annotations.json'
    image_dir_train = cfg.data_dir + 'Data_vcoco/train2014/'

    annotation_val = cfg.data_dir + 'Annotations_vcoco/val_annotations.json'
    image_dir_val = cfg.data_dir + 'Data_vcoco/train2014/'

    annotation_test = cfg.data_dir + 'Annotations_vcoco/test_annotations.json'
    image_dir_test = cfg.data_dir + 'Data_vcoco/val2014/'

    vcoco_train = VcocoDataset(annotation_train, image_dir_train, cfg)
    vcoco_val = VcocoDataset(annotation_val, image_dir_val, cfg)
    vcoco_test = VcocoDataset(annotation_test, image_dir_test, cfg)
    num_workers = cfg.num_work
    dataloader_train = DataLoader(vcoco_train, cfg.batch_size, shuffle=True, collate_fn=vcoco_collate,
                                  num_workers=num_workers, )
    dataloader_val = DataLoader(vcoco_val, cfg.batch_size, shuffle=True, collate_fn=vcoco_collate,
                                num_workers=num_workers, )
    dataloader_test = DataLoader(vcoco_test, 1, shuffle=False, collate_fn=vcoco_collate, num_workers=num_workers, )
    dataloader = {'train': dataloader_train, 'val': dataloader_val, 'test': dataloader_test}
    return dataloader


def run_test(model, data_loader, device, split='test'):
    model.eval()
    true_labels = []
    predicted_scores = []

    person_boxes = []
    object_boxes = []
    img_ids = []
    nums_o_p_infor = []
    class_ids = []
    for image, targets in tqdm(data_loader[split]):
        image = image.to(device)
        person_boxes_, object_boxes_, class_ids_, img_ids_, nums_o_p_info_ = get_side_infos(targets)
        person_boxes.append(person_boxes_)
        object_boxes.append(object_boxes_)
        class_ids.append(class_ids_)

        img_ids += img_ids_
        nums_o_p_infor += nums_o_p_info_

        targets = [t.to(device) for t in targets]
        predicted_hoi, _ = model(image, targets)
        predicted_scores.append(predicted_hoi.data.cpu())
        labels = torch.cat([t.get_field('HOI_labs') for t in targets]).data.cpu()
        true_labels.append(labels)

    predicted_scores = torch.cat(predicted_scores).numpy()
    true_labels = torch.cat(true_labels).numpy()
    ap_results = calculate_averate_precision(predicted_scores, true_labels)
    detections_test = []

    ap_test = pd.DataFrame(ap_results, columns=['TEST', 'Score_TEST'])
    model.train()
    return ap_test, detections_test


def train_test(model, optimizer, scheduler, dataloader, logger, meters, cfg):
    torch.cuda.empty_cache()
    training_phases = ['train', 'val']
    iteration = 0
    mean_best = cfg.mean_best
    end = time.time()
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        scheduler.step()
        for phase in training_phases:
            model.train()
            true_scores = []
            predicted_scores = []
            torch.cuda.empty_cache()
            for image, targets in dataloader[phase]:
                data_time = time.time() - end
                image = image.to(cfg.device)

                targets = [t.to(cfg.device) for t in targets]
                predicted_hoi, loss_dict = model(image, targets)

                losses = sum(loss for loss in loss_dict.values())
                loss_dict_reduced = reduce_loss_dict(loss_dict)
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                meters.update(loss=losses_reduced, **loss_dict_reduced)
                batch_time = time.time() - end
                end = time.time()
                meters.update(time=batch_time, data=data_time)
                eta_seconds = meters.time.global_avg * (100000 - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                optimizer.zero_grad()
                with amp.scale_loss(losses, optimizer) as scaled_losses:
                    scaled_losses.backward()
                clip_grad_norm([(n, p) for n, p in model.named_parameters() if p.requires_grad],
                               max_norm=5, logger=logger, verbose=(iteration % 500 == 0), clip=True)
                optimizer.step()
                if iteration % 50 == 0:
                    logger.info(
                        meters.delimiter.join(
                            [
                                "eta: {eta}",
                                "iter: {iter}",
                                "{meters}",
                                "lr: {lr:.6f}",
                                "max mem: {memory:.0f}"

                            ]
                        ).format(
                            eta=eta_string,
                            iter=iteration,
                            meters=str(meters),
                            lr=optimizer.param_groups[-1]["lr"],
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                        )
                    )

                predicted_scores.append(predicted_hoi)
                true_scores.append(torch.cat([t.get_field('HOI_labs') for t in targets]))
                iteration += 1
            predicted_scores = torch.cat(predicted_scores).data.cpu().numpy()
            true_scores = torch.cat(true_scores).data.cpu().numpy()

            if phase == 'train':
                ap_train = calculate_averate_precision(predicted_scores, true_scores)
                ap_train = pd.DataFrame(ap_train, columns=['Name_TRAIN', 'Score_TRAIN'])
                print(ap_train)

            elif phase == 'val':
                ap_test = calculate_averate_precision(predicted_scores, true_scores)
                ap_test = pd.DataFrame(ap_test, columns=['Name_VAL', 'Score_VAL'])
                print(ap_test)

        if epoch % 3 == 0 and epoch >= 0:
            resutls, detections_test = run_test(model, dataloader, cfg.device)
            print(resutls)

            file_name_p = cfg.output + '/' + 'test{}.pickle'.format(epoch + 1)
            with open(file_name_p, 'wb') as handle:
                pickle.dump(detections_test, handle)

            mean_resutls = resutls.to_records(index=False)[29][1]
            if mean_resutls > mean_best:
                mean_best = mean_resutls
                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'mean_best': mean_best,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, filename=cfg.output + '/' + 'bestcheckpoint.pth.tar')

        if (epoch + 1) % cfg.saving_epoch == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'mean_best': mean_best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, filename=cfg.output + '/model_' + str(epoch + 1).zfill(6) + '_checkpoint.pth.tar')


if __name__ == "__main__":
    main()
