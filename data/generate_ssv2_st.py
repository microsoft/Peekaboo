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
detector = pipeline(model=checkpoint, task="zero-shot-object-detection", cache_dir="/coc/pskynet4/yashjain/", device='cuda:0')


# from transformers import Owlv2Processor, Owlv2ForObjectDetection

# processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
# model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# def owl_inference(image, text):
#     inputs = inputs = processor(text=text, images=image, return_tensors="pt")
#     outputs = model(**inputs)
#     target_sizes = torch.Tensor([image.size[::-1]])
#     results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)
#     return results[0]['boxes']

def find_surrounding_masks(mask_presence):
    # Finds the indices of the surrounding masks for each gap
    gap_info = []
    start = None

    for i, present in enumerate(mask_presence):
        if present and start is not None:
            end = i
            gap_info.append((start, end))
            start = None
        elif not present and start is None and i > 0:
            start = i - 1

    # Handle the special case where the gap is at the end
    if start is not None:
        gap_info.append((start, len(mask_presence)))
    
    return gap_info

def copy_edge_masks(mask_list, mask_presence):
    if not mask_presence[-1]:
        # Find the last present mask and copy it to the end
        for i in reversed(range(len(mask_presence))):
            if mask_presence[i]:
                mask_list[i+1:] = [mask_list[i]] * (len(mask_presence) - i - 1)
                break

def interpolate_masks(mask_list, mask_presence):
    # Ensure the mask list and mask presence list are the same length
    assert len(mask_list) == len(mask_presence), "Mask list and presence list must have the same length."

    # Copy edge masks if there are gaps at the start or end
    # copy_edge_masks(mask_list, mask_presence)

    # Find surrounding masks for gaps
    gap_info = find_surrounding_masks(mask_presence)

    # Interpolate the masks in the gaps
    for start, end in gap_info:
        end = min(end, len(mask_list)-1)
        num_steps = end - start - 1
        prev_mask = mask_list[start]
        next_mask = mask_list[end]
        step = (next_mask - prev_mask) / (num_steps + 1)
        interpolated_masks = [(prev_mask + step * (i + 1)).round().astype(int) for i in range(num_steps)]
        mask_list[start + 1:end] = interpolated_masks

    return mask_list

def get_bounding_boxes(clip_path, subject):
    # Read video from the path
    clip = VideoFileClip(clip_path)
    all_bboxes = []
    bbox_present = []

    num_bb = 0
    
    for fidx,frame in enumerate(clip.iter_frames()):
        if fidx > 24: break

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

    # Design decision
    interpolated_masks = interpolate_masks(all_bboxes, bbox_present)    
    return interpolated_masks, num_bb

import json
BASE_DIR = '/scr/clips_downsampled_5fps_downsized_224x224'
annotations = json.load(open('/gscratch/sewoong/anasery/datasets/ssv2/datasets/SSv2/ssv2_label_ssv2_template/ssv2_ret_label_val_small_filtered.json', 'r'))

records_with_masks = []
ridx = 0
for idx,record in tqdm.tqdm(enumerate(annotations)):
    video_id = record['video']
    print(f"{record['caption']} - {record['nouns']}")
    # for video_id in video_ids:
    new_record = record.copy()
    new_record['video'] = video_id.replace('webm', 'mp4')
    all_masks = []
    all_num_bb = []
    for subject in record['nouns']:
        masks, num_bb = get_bounding_boxes(clip_path=os.path.join(BASE_DIR, video_id.replace('webm', 'mp4')), subject=subject)
        all_masks.append(masks)
        all_num_bb.append(num_bb)
    try:    
        print(f"{record['video']} , subj - {record['nouns']}, bb - {all_num_bb}")
    except:
        continue
    new_record['masks'] = all_masks
    records_with_masks.append(new_record)
    ridx += 1

    if ridx % 100 == 0:
        with open(f'/gscratch/sewoong/anasery/datasets/ssv2/datasets/SSv2/SSv2_label_with_two_obj_masks.pkl', 'wb') as f:
            pickle.dump(records_with_masks, f)

with open(f'/gscratch/sewoong/anasery/datasets/ssv2/datasets/SSv2/SSv2_label_with_two_obj_masks.pkl', 'wb') as f:
    pickle.dump(records_with_masks, f)