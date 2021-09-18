from __future__ import print_function, division
import json
import os
import torch
import pickle
import random
import pandas as pd
import numpy as np
from PIL import Image
import data_preprocess as dp
from skimage import io, transform
from utils import processing_sg, LIS
from transforms import build_transforms
from torchvision import transforms, utils
from structures.bounding_box import BoxList
from torch.utils.data import Dataset, DataLoader
from structures.image_list import to_image_list


# Image list ids that cannot be detected any objects.
bad_detections_train = [6357, 17487, 130851, 153892, 176148, 326601, 364400, 365013, 369213, 429514, 483039, 560726]
bad_detections_val = [130099, 196981, 352877, 387895, 426849, 526087, 567439, 568117]
bad_detections_test = [16875, 30828, 56701, 66706, 75768, 165157, 228418, 230501, 255483, 267725, 293855, 344045,
                       413805, 419143, 451038, 479280, 561411]


def vcoco_collate(batch):
    transposed_batch = list(zip(*batch))
    images = to_image_list(transposed_batch[0], 32)
    targets = transposed_batch[1]
    return images, targets


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        transformed_img = transform.resize(image, (new_h, new_w))
        return transformed_img


class VcocoDataset(Dataset):
    def __init__(self, json_file_image, root_dir, cfg):
        with open(json_file_image) as json_file_:
            self.vcoco_frame_file = json.load(json_file_)
        self.flag = json_file_image.split('/')[-1].split('_')[0]
        self.cfg = cfg
        if self.flag == 'train':
            self.vcoco_frame = [x for x in self.vcoco_frame_file.keys() if x not in str(bad_detections_train)]
            self.detect_results = pickle.load(open(cfg.train_detected_results, 'rb'))
            gt_path = cfg.data_dir + 'Annotations_vcoco/train_annotations.json'
            with open(gt_path) as fp:
                self.annotations = json.load(fp)
        elif self.flag == 'val':
            self.vcoco_frame = [x for x in self.vcoco_frame_file.keys() if x not in str(bad_detections_val)]
            self.detect_results = pickle.load(open(cfg.train_detected_results, 'rb'))
            gt_path = cfg.data_dir + 'Annotations_vcoco/val_annotations.json'
            with open(gt_path) as fp:
                self.annotations = json.load(fp)
        elif self.flag == 'test':
            self.vcoco_frame = [x for x in self.vcoco_frame_file.keys() if x not in str(bad_detections_test)]
            self.detect_results = pickle.load(open(cfg.test_detected_results, 'rb'))
            gt_path = cfg.data_dir + 'Annotations_vcoco/test_annotations.json'
            with open(gt_path) as fp:
                self.annotations = json.load(fp)
        self.root_dir = root_dir
        self.sg_pred = pickle.load(open(cfg.sg_data + self.flag + '_0.4_0.2.pk', 'rb'))
        self.transform = build_transforms(cfg, is_train=(self.flag == 'train'))
        self.max_nagetive = 512

    def __len__(self):
        return len(self.vcoco_frame)

    def convert2target(self, image, res):
        img_info = res['shape']
        w, h = img_info[0], img_info[1]
        box = np.concatenate((res['per_box'], res['obj_box']))
        scales = [w, h, w, h]
        box2 = box / scales
        box = torch.from_numpy(box).clone()
        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        if self.transform is not None:
            image, target = self.transform(image, target)
        target.bbox[res['per_box'].shape[0], :] = 0
        per_obj_labs = np.concatenate((np.array([1 for _ in range(res['per_box'].shape[0])]), res['all_obj_labels']))
        target.add_field("labels", torch.from_numpy(per_obj_labs))
        target.add_field('boxes', torch.from_numpy(box2))
        per_scores = torch.tensor(res['scores_persons']).float()
        per_scores = LIS(per_scores, 8.3, 12, 10)

        obj_scores = torch.tensor(res['scores_objects']).float()
        obj_scores = LIS(obj_scores, 8.3, 12, 10)
        labels_scores = torch.cat((per_scores, obj_scores))

        target.add_field('obj_scores', labels_scores)
        pair_score = []

        pair_info = []
        hoi_labs = res['labels_all']
        target.add_field('hoi_labels', torch.from_numpy(hoi_labs))
        all_labels = target.get_field('labels')[res['labels_all'].shape[0]:]
        obj_label_for_mask = []
        one_hot_labs = []
        num_bg = 0
        for i in range(res['labels_all'].shape[0]):
            for j in range(res['labels_all'].shape[1]):
                if self.flag == 'test':
                    pair_info.append([i, res['labels_all'].shape[0] + j, 1])
                    one_hot_labs.append(target.get_field('hoi_labels')[i, j, :])
                    pair_score.append(per_scores[i] * obj_scores[j])
                    obj_label_for_mask.append(all_labels[j])
                else:
                    if int(res['labels_all'][i, j, :].sum()) > 0:
                        pair_info.append([i, res['labels_all'].shape[0] + j, 1])
                        one_hot_labs.append(target.get_field('hoi_labels')[i, j, :])
                        pair_score.append(per_scores[i] * obj_scores[j])
                        obj_label_for_mask.append(all_labels[j])
                    elif num_bg < self.max_nagetive:
                        pair_info.append([i, res['labels_all'].shape[0] + j, 0])
                        one_hot_labs.append(target.get_field('hoi_labels')[i, j, :])
                        pair_score.append(per_scores[i] * obj_scores[j])
                        num_bg += 1
                        obj_label_for_mask.append(all_labels[j])
                    elif random.random() < 0.5:
                        replace_id = int(random.random() * len(pair_info))
                        pair_info[replace_id] = [i, res['labels_all'].shape[0] + j, 0]
                        one_hot_labs[replace_id] = target.get_field('hoi_labels')[i, j, :]
                        pair_score[replace_id] = per_scores[i] * obj_scores[j]
                        obj_label_for_mask[replace_id] = all_labels[j]

        target.add_field("pairs_info", torch.tensor(pair_info))
        target.add_field("mask", torch.tensor(obj_label_for_mask))
        target.add_field("per_mul_obj_scores", torch.tensor(pair_score))
        target.add_field('HOI_labs', torch.stack(one_hot_labs))

        return image, target

    def __getitem__(self, idx):
        if self.flag == 'test':
            img_pre_suffix = 'COCO_val2014_' + str(self.vcoco_frame[idx]).zfill(12) + '.jpg'
        else:
            img_pre_suffix = 'COCO_train2014_' + str(self.vcoco_frame[idx]).zfill(12) + '.jpg'

        img_name = os.path.join(self.root_dir, img_pre_suffix)
        image = Image.open(img_name).convert('RGB')
        all_info = dp.get_anotation_info_by_imageId(int(self.vcoco_frame[idx]), self.flag, self.detect_results,
                                                    self.annotations)
        image, target = self.convert2target(image, all_info)
        union_box = dp.get_attention_maps(target)
        target.add_field('union_box', torch.tensor(union_box).float())
        target.add_field('sg', self.sg_pred[img_pre_suffix])
        sg_graph = processing_sg(self.sg_pred[img_pre_suffix], target)
        target.add_field('sg_data', sg_graph)
        target.add_field('image_id', self.vcoco_frame[idx])

        return image, target
