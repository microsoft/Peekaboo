#@title Get bounding boxes for the subject
from transformers import pipeline
from moviepy.editor import VideoFileClip
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pickle
import torch



checkpoint = "google/owlvit-large-patch14"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device='cuda:0')


def get_bounding_boxes(clip_path, subject):
    # Read video from the path
    clip = VideoFileClip(clip_path)
    all_bboxes = []
    bbox_present = []

    num_bb = 0
    for fidx,frame in enumerate(clip.iter_frames()):
        frame = Image.fromarray(frame)

        predictions = detector(
            frame,
            candidate_labels=[subject,], 
        )
        try:
            
            bbox = predictions[0]["box"]
            
            bbox = (bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"])
            
            # Get a zeros array of the same size as the frame
            canvas = np.zeros(frame.size[::-1])
            # Draw the bounding box on the canvas
            canvas[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            # Add the canvas to the list of bounding boxes
            all_bboxes.append(canvas)
            bbox_present.append(True)
            num_bb += 1
        except Exception as e:
            
            # Append an empty canvas, we will interpolate later
            all_bboxes.append(np.zeros(frame.size[::-1]))        
            bbox_present.append(False)    
            continue
    return all_bboxes, num_bb

import pickle as pkl
dir_path = '/your/result/path' 

video_filename = '2_of_40_2.mp4'
output_bbox = []
with open("/ssv2dataset/path.pkl", "rb") as f:
    data = pkl.load(f)
    dataset_size = len(data)
    failed_cnt = 0
    for i, d in tqdm.tqdm(enumerate(data)):
        try:
            # print(f"{d['subject']} || {d['caption']} || {d['video']}")
            filename = d['id'] 
            video_path = os.path.join(dir_path, filename, video_filename)
            fg_object = d['subject']
            masks, num_bb = get_bounding_boxes(video_path, fg_object)

            output_bbox.append({   
                'caption': d['caption'],
                'id': d['id'],
                'subject': d['subject'],
                'mask': masks,
                'num_bb': num_bb
            })
            # print(num_bb)
        except:
            print(f"Missed #{i} with Caption: {d['caption']}")
            failed_cnt += 1

with open(f"/output/path/iou_eval/ssv2_modelscope_{video_filename.split('.')[0]}_bbox-v2.pkl", "wb") as f:
    pkl.dump(output_bbox, f)

print(f"Failed: {failed_cnt} out of {dataset_size}")#@title Get bounding boxes for the subject


checkpoint = "google/owlvit-large-patch14"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection", device='cuda:0')


def get_bounding_boxes(clip_path, subject):
    # Read video from the path
    clip = VideoFileClip(clip_path)
    all_bboxes = []
    bbox_present = []

    num_bb = 0
    for fidx,frame in enumerate(clip.iter_frames()):
        frame = Image.fromarray(frame)

        predictions = detector(
            frame,
            candidate_labels=[subject,], 
        )
        try:
            
            bbox = predictions[0]["box"]
            
            bbox = (bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"])
            
            # Get a zeros array of the same size as the frame
            canvas = np.zeros(frame.size[::-1])
            # Draw the bounding box on the canvas
            canvas[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            # Add the canvas to the list of bounding boxes
            all_bboxes.append(canvas)
            bbox_present.append(True)
            num_bb += 1
        except Exception as e:
            
            # Append an empty canvas, we will interpolate later
            all_bboxes.append(np.zeros(frame.size[::-1]))        
            bbox_present.append(False)    
            continue
    return all_bboxes, num_bb

import pickle as pkl
dir_path = '/your/result/path' 

video_filename = '2_of_40_2.mp4'
output_bbox = []
with open("/ssv2dataset/path.pkl", "rb") as f:
    data = pkl.load(f)
    dataset_size = len(data)
    failed_cnt = 0
    for i, d in tqdm.tqdm(enumerate(data)):
        try:
            # print(f"{d['subject']} || {d['caption']} || {d['video']}")
            filename = d['video'].split('.')[0] 
            video_path = os.path.join(dir_path, filename, video_filename)
            fg_object = d['subject']
            masks, num_bb = get_bounding_boxes(video_path, fg_object)

            output_bbox.append({   
                'caption': d['caption'],
                'video': d['video'],
                'subject': d['subject'],
                'mask': masks,
                'num_bb': num_bb
            })
            # print(num_bb)
        except:
            print(f"Missed #{i} with Caption: {d['caption']}")
            failed_cnt += 1

with open(f"/output/path/iou_eval/ssv2_modelscope_{video_filename.split('.')[0]}_bbox-v2.pkl", "wb") as f:
    pkl.dump(output_bbox, f)

print(f"Failed: {failed_cnt} out of {dataset_size}")