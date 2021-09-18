import json
import numpy as np
import argparse
from random import randint
import cv2
import torch
import pickle
from torchvision.ops import nms

VERB2ID = {u'carry': 0,
           u'catch': 1,
           u'cut_instr': 2,
           u'cut_obj': 3,
           u'drink': 4,
           u'eat_instr': 5,
           u'eat_obj': 6,
           u'hit_instr': 7,
           u'hit_obj': 8,
           u'hold': 9,
           u'jump': 10,
           u'kick': 11,
           u'lay': 12,
           u'look': 13,
           u'point': 14,
           u'read': 15,
           u'ride': 16,
           u'run': 17,
           u'sit': 18,
           u'skateboard': 19,
           u'ski': 20,
           u'smile': 21,
           u'snowboard': 22,
           u'stand': 23,
           u'surf': 24,
           u'talk_on_phone': 25,
           u'throw': 26,
           u'walk': 27,
           u'work_on_computer': 28
           }
NO_VERBS = 29

def get_detections_from_memory(segment_key, flag, detected_data, annotations):
    SCORE_TH = 0.6
    SCORE_OBJ = 0.3
    if flag == 'train':
        annotation = annotations[str(segment_key)]
        cur_obj_path_s = "COCO_train2014_%.12i.json" % (segment_key)
    elif flag == 'test':
        annotation = annotations[str(segment_key)]
        cur_obj_path_s = "COCO_val2014_%.12i.json" % (segment_key)
    elif flag == 'val':
        annotation = annotations[str(segment_key)]
        cur_obj_path_s = "COCO_train2014_%.12i.json" % (segment_key)

    annotation = clean_up_annotation(annotation)
    detections = detected_data[cur_obj_path_s]

    img_H = detections['H']
    img_W = detections['W']
    shape = [img_W, img_H]
    persons_d, objects_d = filter_bad_detections(detections, SCORE_TH, SCORE_OBJ)
    d_p_boxes, scores_persons, class_id_humans = get_boxes_det(persons_d, img_H, img_W)
    d_o_boxes, scores_objects, class_id_objects = get_boxes_det(objects_d, img_H, img_W)
    scores_objects.insert(0, 1)
    return d_p_boxes, d_o_boxes, scores_persons, scores_objects, class_id_humans, class_id_objects, annotation, shape



def get_attention_maps(tg):
    boxes = tg.get_field('boxes')
    no_person_dets = tg.get_field('hoi_labels').shape[0]
    persons_np = boxes[:no_person_dets]
    objects_np = boxes[no_person_dets:]
    labs = tg.get_field('mask')
    union_box = []
    pairs = tg.get_field('pairs_info')
    for cnt, pair in enumerate(pairs):
        dd_i = pair[0]
        do_i = pair[1] - no_person_dets
        union_box.append(union_BOX(persons_np[dd_i], objects_np[do_i], obj_id=labs[cnt]))
    return torch.from_numpy(np.concatenate((union_box)))


def get_anotation_info_by_imageId(segment_key, flag, detected_data, annotations):
    d_p_boxes, d_o_boxes, scores_persons, scores_objects, class_id_humans, class_id_objects, annotation, shape = get_detections_from_memory(
        segment_key, flag, detected_data, annotations)
    if flag == 'test':
        MATCHING_IOU = 0.5
    else:
        MATCHING_IOU = 0.4
    no_person_dets = len(d_p_boxes)
    no_object_dets = len(d_o_boxes)
    labels_np = np.zeros([no_person_dets, no_object_dets + 1, NO_VERBS], np.int32)
    all_obj_classes = np.array([1] + class_id_objects)
    a_p_boxes = [ann['person_box'] for ann in annotation]
    iou_mtx = get_iou_mtx(a_p_boxes, d_p_boxes)
    d_o_boxes_tensor = np.array([[0, 0, 0, 0]] + d_o_boxes, np.float32)
    d_p_boxes_tensor = np.array(d_p_boxes, np.float32)
    if no_person_dets != 0 and len(a_p_boxes) != 0:
        max_iou_for_each_det = np.max(iou_mtx, axis=0)
        index_for_each_det = np.argmax(iou_mtx, axis=0)
        for dd in range(no_person_dets):
            cur_max_iou = max_iou_for_each_det[dd]
            if cur_max_iou < MATCHING_IOU:
                continue
            matched_ann = annotation[index_for_each_det[dd]]
            hoi_anns = matched_ann['hois']
            noobject_hois = [oi for oi in hoi_anns if len(oi['obj_box']) == 0]

            for no_hoi in noobject_hois:
                verb_idx = VERB2ID[no_hoi['verb']]
                labels_np[dd, 0, verb_idx] = 1

            object_hois = [oi for oi in hoi_anns if len(oi['obj_box']) != 0]
            a_o_boxes = [oi['obj_box'] for oi in object_hois]
            iou_mtx_o = get_iou_mtx(a_o_boxes, d_o_boxes)

            if a_o_boxes and d_o_boxes:
                for do in range(len(d_o_boxes)):
                    for ao in range(len(a_o_boxes)):
                        cur_iou = iou_mtx_o[ao, do]
                        # enough iou
                        if cur_iou < MATCHING_IOU:
                            continue
                        current_hoi = object_hois[ao]
                        verb_idx = VERB2ID[current_hoi['verb']]
                        labels_np[dd, do + 1, verb_idx] = 1  # +1 because 0 is no object

        comp_labels = labels_np.reshape(no_person_dets * (no_object_dets + 1), NO_VERBS)
        labels_single = np.array([1 if i.any() == True else 0 for i in comp_labels])
        labels_single = labels_single.reshape(np.shape(labels_single)[0], 1)
        return {'per_box': d_p_boxes_tensor, "obj_box": d_o_boxes_tensor, 'labels_all': labels_np,
                'info': (len(d_p_boxes_tensor), len(d_o_boxes_tensor)), 'labels_single': labels_single,
                'all_obj_labels': all_obj_classes, 'scores_persons': scores_persons, 'scores_objects': scores_objects,
                'shape': shape}
    else:
        comp_labels = labels_np.reshape(no_person_dets * (no_object_dets + 1), NO_VERBS)
        labels_single = np.array([1 if i.any() == True else 0 for i in comp_labels])
        labels_single = labels_single.reshape(np.shape(labels_single)[0], 1)
        return {'per_box': d_p_boxes_tensor, "obj_box": d_o_boxes_tensor, 'labels_all': labels_np,
                'all_obj_labels': all_obj_classes,
                'scores_persons': scores_persons, 'scores_objects': scores_objects, 'labels_single': labels_single,
                'info': (len(d_p_boxes_tensor), len(d_o_boxes_tensor)), 'shape': shape}



def union_BOX(roi_pers, roi_objs, H=64, W=64, obj_id=0):
    assert H == W
    roi_pers = np.array(roi_pers * H, dtype=int)
    roi_objs = np.array(roi_objs * H, dtype=int)
    sample_box = np.zeros([1, 2, H, W])
    sample_box[0, 0, roi_pers[1]:roi_pers[3] + 1, roi_pers[0]:roi_pers[2] + 1] = 1
    sample_box[0, 1, roi_objs[1]:roi_objs[3] + 1, roi_objs[0]:roi_objs[2] + 1] = obj_id
    return sample_box


def clean_up_annotation(annotation):
    persons_dict = {}
    for hoi in annotation:
        box = hoi['person_bbx']
        box = [int(coord) for coord in box]
        dkey = tuple(box)
        objects = hoi['object']
        if len(objects['obj_bbx']) == 0:  # no obj case
            cur_oi = {'verb': hoi['Verbs'],
                      'obj_box': [],
                      # 'obj_str': '',
                      }
        else:
            cur_oi = {'verb': hoi['Verbs'],
                      'obj_box': [int(coord) for coord in hoi['object']['obj_bbx']],
                      # 'obj_str': hoi['object']['obj_name'],
                      }

        if dkey in persons_dict:
            persons_dict[dkey]['hois'].append(cur_oi)
        else:
            persons_dict[dkey] = {'person_box': box, 'hois': [cur_oi]}

    pers_list = []
    for dkey in persons_dict:
        pers_list.append(persons_dict[dkey])

    return pers_list


def get_boxes_det(dets, img_H, img_W):
    boxes = []
    scores = []
    class_no = []
    for det in dets:
        top, left, bottom, right = det['box_coords']
        scores.append(det['score'])
        class_no.append(det['class_no'])
        left, top, right, bottom = left * img_W, top * img_H, right * img_W, bottom * img_H
        # left, top, right, bottom = left, top, right, bottom
        boxes.append([left, top, right, bottom])
    return boxes, scores, class_no


def get_iou_mtx(anns, dets):
    no_gt = len(anns)
    no_dt = len(dets)
    iou_mtx = np.zeros([no_gt, no_dt])

    for gg in range(no_gt):
        gt_box = anns[gg]
        for dd in range(no_dt):
            dt_box = dets[dd]
            iou_mtx[gg, dd] = IoU_box(gt_box, dt_box)

    return iou_mtx


def filter_bad_detections(detections, SCORE_TH, SCORE_OBJ):
    persons = []
    objects = []
    for det in detections['detections']:
        if det['class_str'] == 'person':
            if det['score'] < SCORE_TH:
                continue
            persons.append(det)
        else:
            if det['score'] < SCORE_OBJ:
                continue
            objects.append(det)

    return persons, objects

def IoU_box(box1, box2):
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    left_int = max(left1, left2)
    top_int = max(top1, top2)
    right_int = min(right1, right2)
    bottom_int = min(bottom1, bottom2)
    areaIntersection = max(0, right_int - left_int) * max(0, bottom_int - top_int)
    area1 = (right1 - left1) * (bottom1 - top1)
    area2 = (right2 - left2) * (bottom2 - top2)
    IoU = areaIntersection / float(area1 + area2 - areaIntersection)
    return IoU
