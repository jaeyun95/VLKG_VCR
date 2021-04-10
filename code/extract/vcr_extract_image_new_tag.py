"""
Dataloaders for VCR
"""
import json
import os
import h5py
import numpy as np
import torch
from config import VCR_IMAGES_DIR, VCR_ANNOTS_DIR
from dataloaders.vcr_attribute_new_tag import VCR, VCRLoader

################################################
#this is for vcr attribute dataset
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

print('detector load ok!')

################################################
#define saving file name!
################################################
output_h5 = h5py.File(f'extract_feature/vcr_new_tag_image_val.h5', 'w')
output_h5_2 = h5py.File(f'extract_feature/vcr_new_tag_image_train.h5', 'w')
################################################
#start now!
################################################
check_list = []
print('val start!')
for i,item in enumerate(val_loader):
    print(i)
    image_id = item['metadata'][0]['img_id'].split('-')[1]
    if image_id not in check_list:
        check_list.append(image_id)
    else: continue
    tag_feature_path = os.path.join('/YOUR_PATH/',
                                         f'attribute_features_val_scene_version.h5')
    non_tag_feature_path = os.path.join('/YOUR_PATH/',
                                        f'new_tag_features_val_scene_version.h5')

    with h5py.File(tag_feature_path, 'r') as h5:
        tag_features = np.array(h5[str(image_id)]['features'], dtype=np.float32)
        tag_boxes = np.array(h5[str(image_id)]['boxes'], dtype=np.float32)

    with h5py.File(non_tag_feature_path, 'r') as h5:
        non_tag_boxes = np.array(h5[str(image_id)]['boxes'], dtype=np.float32)
        non_tag_features = np.array(h5[str(image_id)]['features'], dtype=np.float32)


    images = item['images'][0]
    image_h = images.shape[1]
    image_w = images.shape[2]

    final_boxes = np.concatenate((tag_boxes, non_tag_boxes))
    final_features = np.concatenate((tag_features, non_tag_features))

    #features = torch.squeeze(tag_features)
    #print('boxes1 ',item['boxes'])
    #boxes = torch.squeeze(item['boxes'])
    #print('boxes ', boxes)
    #num_boxes = boxes.shape[0]
    #if (num_boxes) != tag_boxes.shape[0]:
    #   print('val!!!!! different!!!!')
    #print('image_h ', image_h)
    #print('image_w ', image_w)
    #print('num_boxes ', num_boxes)

    #print(len(item['boxes']))
    #print(len(obj_reps['obj_reps']))

    output_h5.create_group(f'{image_id}')
    group2use = output_h5[f'{image_id}']
    group2use.create_dataset(f'image_id', data=image_id)
    group2use.create_dataset(f'image_h', data=image_h)
    group2use.create_dataset(f'image_w', data=image_w)
    group2use.create_dataset(f'features', data=final_features)
    group2use.create_dataset(f'boxes', data=final_boxes)
    group2use.create_dataset(f'num_boxes', data=final_boxes.shape[0])

check_list = []
print('train start!')
for i, item in enumerate(train_loader):
        print(i)
        image_id = item['metadata'][0]['img_id'].split('-')[1]
        if image_id not in check_list:
            check_list.append(image_id)
        else:
            continue
        images = item['images'][0]
        image_h = images.shape[1]
        image_w = images.shape[2]
        #features = torch.squeeze(item['det_features'])
        # print('boxes1 ',item['boxes'])

        # print('boxes ', boxes)

        tag_feature_path = os.path.join('/YOUR_PATH/',
                                        f'attribute_features_train_scene_version.h5')
        non_tag_feature_path = os.path.join('/YOUR_PATH/',
                                            f'new_tag_features_train_scene_version.h5')

        with h5py.File(tag_feature_path, 'r') as h5:
            tag_features = np.array(h5[str(image_id)]['features'], dtype=np.float32)
            tag_boxes = np.array(h5[str(image_id)]['boxes'], dtype=np.float32)

        with h5py.File(non_tag_feature_path, 'r') as h5:
            non_tag_boxes = np.array(h5[str(image_id)]['boxes'], dtype=np.float32)
            non_tag_features = np.array(h5[str(image_id)]['features'], dtype=np.float32)
        # print('image_h ', image_h)
        # print('image_w ', image_w)
        # print('num_boxes ', num_boxes)

        # print(len(item['boxes']))
        # print(len(obj_reps['obj_reps']))
        final_boxes = np.concatenate((tag_boxes, non_tag_boxes))
        final_features = np.concatenate((tag_features, non_tag_features))

        output_h5_2.create_group(f'{image_id}')
        group2use = output_h5_2[f'{image_id}']
        group2use.create_dataset(f'image_id', data=image_id)
        group2use.create_dataset(f'image_h', data=image_h)
        group2use.create_dataset(f'image_w', data=image_w)
        group2use.create_dataset(f'features', data=final_features)
        group2use.create_dataset(f'boxes', data=final_boxes)
        group2use.create_dataset(f'num_boxes', data=final_boxes.shape[0])
