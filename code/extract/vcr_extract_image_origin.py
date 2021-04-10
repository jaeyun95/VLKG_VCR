"""
Dataloaders for VCR
"""
import json
import os
import h5py
import numpy as np
import torch
from utils.detector_101 import SimpleDetector
from config import VCR_IMAGES_DIR, VCR_ANNOTS_DIR
from dataloaders.vcr import VCR, VCRLoader


################################################
#this is for vcr dataset
################################################


################################################
#data splits!
################################################
train, val = VCR.splits(mode='rationale', embs_to_load='bert_da', only_use_relevant_dets=False)

print('split is ok!')


################################################
#data loader!
################################################
loader_params = {'batch_size': 1, 'num_gpus':1, 'num_workers':1}
train_loader = VCRLoader.from_dataset(train, **loader_params)
val_loader = VCRLoader.from_dataset(val, **loader_params)

print('loader is ok!')

################################################
#define detector!
################################################
detector = SimpleDetector(pretrained=True, average_pool=True, semantic=True, final_dim=2048)

print('detector load ok!')

################################################
#define saving file name!
################################################
output_h5_val = h5py.File(f'extract_feature/vcr_image_val.h5', 'w')
output_h5 = h5py.File(f'extract_feature/vcr_image_train.h5', 'w')

################################################
#start now!
################################################
check_list = []
for i,item in enumerate(val_loader):
    print(i)
    image_id = item['metadata'][0]['img_id'].split('-')[1]
    if image_id not in check_list:
        check_list.append(image_id)
    else: continue
    obj_reps = detector(images=item['images'], boxes=item['boxes'], box_mask=item['box_mask'], classes=item['objects'], segms=item['segms'])
    images = torch.squeeze(item['images'])
    image_h = images.shape[1]
    image_w = images.shape[2]
    features = torch.squeeze(obj_reps['obj_reps'])
    #print('boxes1 ',item['boxes'])
    boxes = torch.squeeze(item['boxes'])
    #print('boxes ', boxes)
    num_boxes = boxes.shape[0]

    #print('image_h ', image_h)
    #print('image_w ', image_w)
    #print('num_boxes ', num_boxes)

    #print(len(item['boxes']))
    #print(len(obj_reps['obj_reps']))
    output_h5_val.create_group(f'{image_id}')
    group2use = output_h5_val[f'{image_id}']
    group2use.create_dataset(f'image_id', data=image_id)
    group2use.create_dataset(f'image_h', data=image_h)
    group2use.create_dataset(f'image_w', data=image_w)
    group2use.create_dataset(f'features', data=features.detach())
    group2use.create_dataset(f'boxes', data=boxes.detach())
    group2use.create_dataset(f'num_boxes', data=num_boxes)

check_list = []
for i, item in enumerate(train_loader):
        print(i)
        image_id = item['metadata'][0]['img_id'].split('-')[1]
        if image_id not in check_list:
            check_list.append(image_id)
        else:
            continue
        obj_reps = detector(images=item['images'], boxes=item['boxes'], box_mask=item['box_mask'],
                            classes=item['objects'], segms=item['segms'])
        images = torch.squeeze(item['images'])
        image_h = images.shape[1]
        image_w = images.shape[2]
        features = torch.squeeze(obj_reps['obj_reps'])
        # print('boxes1 ',item['boxes'])
        boxes = torch.squeeze(item['boxes'])
        # print('boxes ', boxes)
        num_boxes = boxes.shape[0]

        # print('image_h ', image_h)
        # print('image_w ', image_w)
        # print('num_boxes ', num_boxes)

        # print(len(item['boxes']))
        # print(len(obj_reps['obj_reps']))
        output_h5.create_group(f'{image_id}')
        group2use = output_h5[f'{image_id}']
        group2use.create_dataset(f'image_id', data=image_id)
        group2use.create_dataset(f'image_h', data=image_h)
        group2use.create_dataset(f'image_w', data=image_w)
        group2use.create_dataset(f'features', data=features.detach())
        group2use.create_dataset(f'boxes', data=boxes.detach())
        group2use.create_dataset(f'num_boxes', data=num_boxes)
