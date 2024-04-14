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
import argparse


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
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_filename', type=str, default='2_of_40_2.mp4', help='Name format of the videos generated')
    parser.add_argument('--generated_directory', type=str, help='Path to the directory containing generated videos')
    parser.add_argument('--gt_file', type=str, help='Path to the ground truth annotations')
    
    args = parser.parse_args()
    
    dir_path = args.generated_directory 

    video_filename = args.video_filename
    output_bbox = []
    with open(args.gt_file, "r") as f:
        data = json.load(f.read())
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

    with open(f"{dir_path}/{video_filename.split('.')[0]}_bbox.pkl", "wb") as f:
        pkl.dump(output_bbox, f)

    print(f"Failed: {failed_cnt} out of {dataset_size}")
