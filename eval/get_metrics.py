# Get all metrics for your predicted and ground truth bounding boxes

import pickle as pkl
import json
import numpy as np
import os
import torch
import torch.nn.functional as F

import argparse

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / (np.sum(union) + 1e-5)
    return iou_score

# write a function to calculate AP50
def calculate_ap50(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    # print(intersection.shape, union.shape)
    iou_score = np.sum(intersection, axis=(-1,-2)) / (np.sum(union, axis=(-1,-2)) + 1e-5)
 
    non_zero_prediction_mask = np.sum(mask2, axis=(-1,-2)) > 0
 
    iou_score = iou_score[non_zero_prediction_mask]
    ap50 = np.where(iou_score >= 0.5, 1, 0)
    return np.mean(ap50)

def get_centroid(mask):
    x_argmax = np.argmax(np.sum(mask, axis=(-1)), axis=-1)
    y_argmax = np.argmax(np.sum(mask, axis=(-2)), axis=-1)
    x_rev_argmax = np.argmax(np.cumsum(np.sum(mask, axis=(-1)),axis=-1), axis=-1)
    y_rev_argmax = np.argmax(np.cumsum(np.sum(mask, axis=(-2)),axis=-1), axis=-1)
    cent_x = (x_argmax + x_rev_argmax) / 2
    cent_y = (y_argmax + y_rev_argmax) / 2
    
    centroid = np.stack([cent_x, cent_y], axis=1)
    return centroid
    # return np.linalg.norm(centroid - actual_centr) #+np.linalg.norm(288-cent_y[0])) / 2
    # np.abs(170-cent_x[0])+np.abs(288-cent_y[0])

def calculate_centroid_distance(mask1, mask2):
    c1 = get_centroid(mask1)
    c2 = get_centroid(mask2)
    dist = np.linalg.norm((c1 - c2), ord=2, axis=-1).mean()
    return dist

def mask_size(mask):
    return np.sqrt(np.sum(mask))
def mask_variance(masks):
    mask_sizes = [mask_size(x) for x in masks]
    mask_sizes = np.array(mask_sizes)
    normalized_variance = np.std(mask_sizes) / (np.mean(mask_sizes)+1e-15)
    return normalized_variance

def get_mask_from_bb(bboxes, shape):
    all_bboxes = []
    bbox_present = []
    num_bb = 0
    for bbox in bboxes:
        try:
            # Get a zeros array of the same size as the frame
            canvas = np.zeros(shape)
            # Draw the bounding box on the canvas
            canvas[int(bbox[1]*shape[0]):int(bbox[3]*shape[0]), int(bbox[0]*shape[1]):int(bbox[2]*shape[1])] = 1
            # Add the canvas to the list of bounding boxes
            all_bboxes.append(canvas)
            bbox_present.append(True)
            num_bb += 1
        except Exception as e:
            # Append an empty canvas, we will interpolate later
            all_bboxes.append(np.zeros(shape))        
            bbox_present.append(False)    
            continue
    return all_bboxes

    
def process(data, gt_ds, verbose=False):
    new_data = []

    for rec in data:
        new_rec = {}
        found = False
        for v in gt_ds:
            orig_id = str(v['id'])
            if str(orig_id) == str(rec['id']): 
                found = True
                break
        if not found:
            print(f"Could not find {rec['id']}")
            continue
        rec['frames'] = get_mask_from_bb(v['bboxes'], rec['mask'][0].shape)
            
        if rec['frames'][0].shape != rec['mask'][0].shape:
            rec['frames'] = F.interpolate(torch.tensor(rec['frames']).unsqueeze(0).float(), size=(rec['mask'][0].shape[0], rec['mask'][0].shape[1]), mode='bilinear', align_corners=True).squeeze(0).round().numpy()
        if len(rec['frames']) != len(rec['mask']):
            min_len = min(len(rec['mask']), len(rec['frames']))
            rec['frames'] = rec['frames'][:min_len]
            rec['mask'] = rec['mask'][:min_len]
        rec['iou'] = calculate_iou(rec['frames'], rec['mask'])
        rec['ap50'] = calculate_ap50(rec['frames'], rec['mask'])
        rec['centroid_dist'] = calculate_centroid_distance(rec['frames'], rec['mask'])
        rec['gt_area'] = np.sum(rec['frames'])
        rec['pred_area'] = np.sum(rec['mask'])
        new_rec['mask_variance'] = mask_variance(rec['frames'])
        if rec['gt_area'] < 5: continue
        for key in rec:
            if 'frame' in key or 'mask' in key: continue
            new_rec[key] = rec[key]
        if verbose:
            print(f"Subject: {rec['subject']}, mIoU: {rec['iou']:.2f}, AP50: {rec['ap50']:.2f}, CD: {rec['centroid_dist']:.2f}, GT Area: {rec['gt_area']:.2f}, Pred Area: {rec['pred_area']:.2f}, Num BB: {rec['num_bb']}")
        new_data.append(new_rec)

    return new_data


def print_metrics(new_data):
    df = pd.DataFrame(new_data)
    df['subject'] = df['subject'].str.lower().str.strip()
    df_filtered = df
    df_filtered['area_ratio'] = abs(df_filtered['gt_area']-df_filtered['pred_area']) / df_filtered['gt_area']
    df_filtered['dist'] = df_filtered['centroid_dist'] / (np.linalg.norm([320, 576]))
    df_filtered['ap_50_num_bb'] = df_filtered['ap50'] * df_filtered['num_bb']
    # Create a new column which is a multiplication of iou and (num_bb > 12)
    df_filtered['iou_numbb'] = df_filtered['iou'] * (df_filtered['num_bb'] > 8)
    df_filtered['num_bb_cov'] = (df_filtered['num_bb'] > 8)



    df_filtered['area_ratio'] = abs(df_filtered['gt_area']-df_filtered['pred_area']) / df_filtered['gt_area']
    df_filtered['dist'] = df_filtered['centroid_dist'] / (np.linalg.norm([320, 576]))
    print(f"mIoU , Coverage , CD , AP50")
    print(f"{(df_filtered['iou_numbb'].sum() / (df_filtered['num_bb_cov'].sum()))*100:.1f} , {df_filtered['num_bb_cov'].mean()} , {df_filtered['dist'].mean():.2f} , {df_filtered['ap_50_num_bb'].sum() / df_filtered['num_bb'].sum() * 100:.1f}")
    print(f"{(df_filtered['iou'].mean())*100:.1f} , {df_filtered['num_bb_cov'].mean()} , {df_filtered['dist'].mean():.2f} , {df_filtered['ap50'].mean() * 100:.1f}")
    print("-"*10)    
    
    
def main(pred_path, gt_path):
    with open(pred_path, 'rb') as f:
        data = pkl.load(f)
    with open(gt_path, 'r') as f:
        gt_ds = json.load(f.read())
    new_data = process(data, gt_ds)
    print_metrics(new_data)
    return new_data

if __name__ == "__main__":
    import pandas as pd
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_path', type=str, help="Path to your predicted bounding boxes pickle", required=True)
    parser.add_argument('--gt_path', type=str, help="Path to your ground truth bounding boxes json", required=True)
    args = parser.parse_args()
    print("Starting evaluation")
    main(args.prediction_path, args.gt_path)
