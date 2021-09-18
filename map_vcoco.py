import sys
import json
import vsrl_utils as vu
import numpy as np
import argparse
from vsrl_eval import VCOCOeval
import pickle

sys.path.insert(0, all_data_dir + 'v-coco/')

parser = argparse.ArgumentParser()
parser.add_argument('-sa', '--saving_epoch', type=int, required=False, default=5)
parser.add_argument('-fw', '--first_word', type=str, required=False, default='result')
parser.add_argument('-t', '--types_of_data', type=str, required=False, default='train')
args = parser.parse_args()
saving_epoch = args.saving_epoch
first_word = args.first_word
flag = args.types_of_data
folder_name = '{}'.format(first_word)

if flag == 'train':
    vsrl_annot_file_s = path + '/data/vcoco/vcoco_train.json'
    split_file_s = path + '/data/splits/vcoco_train.ids'

elif flag == 'test':
    vsrl_annot_file_s = path + '/data/vcoco/vcoco_test.json'
    split_file_s = path + '/data/splits/vcoco_test.ids'

elif flag == 'val':
    vsrl_annot_file_s = path + '/data/vcoco/vcoco_val.json'
    split_file_s = path + '/data/splits/vcoco_val.ids'

coco_file_s = path + '/data/instances_vcoco_all_2014.json'
vcocoeval = VCOCOeval(vsrl_annot_file_s, coco_file_s, split_file_s)
file_name = '../' + folder_name + '/' + '{}{}.pickle'.format(flag, saving_epoch)
with open(file_name, 'rb') as handle:
    b = pickle.load(handle)
vcocoeval._do_eval(b, ovr_thresh=0.5)
