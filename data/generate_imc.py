import json
import numpy as np
import random
import csv
import pickle
import tqdm

def clamp(x, min_val, max_val):
    return int(max(min(x, max_val), min_val))

def generate_moving_frames_simpler(canvas_size, num_frames, aspect_ratio, bounding_box_size, motion_type, up_to_down_strict=False, keep_in_frame=True):
    # Mapping size to bounding box dimensions
    size_mapping = {'Small': 0.25, 'Medium': 0.3, 'Large': 0.3}
    aspect_ratio_mapping = {'Rectangle Vertical': (1.33, 1), 'Rectangle Horizontal': (1, 1.33), 'Square': (1, 1)}

    # Calculate bounding box dimensions
    ratio = aspect_ratio_mapping[aspect_ratio]
    box_height = int(canvas_size[0] * size_mapping[bounding_box_size] * ratio[0])
    box_width = int(canvas_size[1] * size_mapping[bounding_box_size] * ratio[1])

    x_init_pos = [0.1 * canvas_size[1], 0.25 * canvas_size[1], 0.45*canvas_size[1], 0.7 * canvas_size[1]]
    y_init_pos = [0.1 * canvas_size[0], 0.25 * canvas_size[0], 0.45*canvas_size[0], 0.7 * canvas_size[0]]
    
    speed_dir = random.choice([-1,1]) # random.randint(1, 3)*4
    # print('-'*20)
    # print(motion_type)
    if 'up' in motion_type.lower():
        # Freedom in horizontal init
        # Vertical init depends on upward or downward motion
        pos_x = random.choice(x_init_pos) + random.randint(int(-0.01 * canvas_size[1]), int(0.01 * canvas_size[1]))
        if up_to_down_strict == 'up':
            # pos_y = np.random.choice(y_init_pos[2:]) + random.randint(int(-0.01 * canvas_size[1]), int(0.01 * canvas_size[1]))
            speed_dir = -1.
        elif up_to_down_strict == 'down':
            # pos_y = np.random.choice(y_init_pos[2:]) + random.randint(int(-0.01 * canvas_size[1]), int(0.01 * canvas_size[1]))
            speed_dir = 1.            
            # y_end_max = canvas_size[0] - box_height 
            
        # else:
        if speed_dir == 1.:
            pos_y = np.random.choice(y_init_pos[:2]) + random.randint(int(-0.01 * canvas_size[0]), int(0.01 * canvas_size[0]))
            y_end_max = canvas_size[0] - box_height
        else:
            pos_y = np.random.choice(y_init_pos[2:]) + random.randint(int(-0.01 * canvas_size[0]), int(0.01 * canvas_size[0]))
            y_end_max = box_height 
        max_speed = np.abs(y_end_max - pos_y) / num_frames
        
        speed = random.randint(2, 4)*4
        speed = min(speed, max_speed)
        speed = speed_dir * speed
    elif 'left' in motion_type.lower():
        # Freedom in vertical init
        # Horizontal init depends on upward or downward motion
        pos_y = random.choice(y_init_pos) + random.randint(int(-0.01 * canvas_size[0]), int(0.01 * canvas_size[0]))
        if up_to_down_strict:
            speed_dir = 1.
            
        if speed_dir == 1.:
            pos_x = np.random.choice(x_init_pos[:2]) + random.randint(int(-0.01 * canvas_size[1]), int(0.01 * canvas_size[1]))
            x_end_max = canvas_size[1] - box_width
        else:
            pos_x = np.random.choice(x_init_pos[2:]) + random.randint(int(-0.01 * canvas_size[1]), int(0.01 * canvas_size[1]))
            x_end_max = box_width 
        max_speed = np.abs(x_end_max - pos_x) / num_frames
        
        speed = random.randint(2, 4)*4
        speed = min(speed, max_speed)
        speed = speed_dir * speed

    else:
        speed_dir_y = random.choice([-1,1]) 
        if speed_dir == 1.:
            pos_x = np.random.choice(x_init_pos[:2]) + random.randint(int(-0.01 * canvas_size[1]), int(0.01 * canvas_size[1]))
            x_end_max = canvas_size[1] - box_width
        else:
            pos_x = np.random.choice(x_init_pos[2:]) + random.randint(int(-0.01 * canvas_size[1]), int(0.01 * canvas_size[1]))
            x_end_max = box_width 

        if speed_dir_y == 1.:
            pos_y = np.random.choice(y_init_pos[:2]) + random.randint(int(-0.01 * canvas_size[0]), int(0.01 * canvas_size[0]))
            y_end_max = canvas_size[0] - box_height
        else:
            pos_y = np.random.choice(y_init_pos[2:]) + random.randint(int(-0.01 * canvas_size[0]), int(0.01 * canvas_size[0]))
            y_end_max = box_height                     
        max_speed_x = np.abs(x_end_max - pos_x) / num_frames
        max_speed_y = np.abs(y_end_max - pos_y) / num_frames
        speed_x = random.randint(2, 4)*4
        speed_y = random.randint(2, 4)*4
        speed_x = min(speed_x, max_speed_x)
        speed_y = min(speed_y, max_speed_y)
        speed_x, speed_y = (speed_dir * speed_x, speed_dir_y * speed_y)

    frames = []

    
    for _ in range(num_frames):
        canvas = np.zeros(canvas_size)

        # Determine movement direction and apply movement
        if motion_type == "Left to right":
            pos_x = (pos_x + speed) # % (canvas_size[1] - box_width)
            pos_y = pos_y + random.randint(int(-0.01 * canvas_size[0]), int(0.01 * canvas_size[0]))
        elif motion_type == "Up to down":
            pos_y = (pos_y + speed) # % (canvas_size[0] - box_height)
            pos_x = pos_x + random.randint(int(-0.01 * canvas_size[1]), int(0.01 * canvas_size[1]))
        elif motion_type == "Zig-zag":
            # Zig-zag motion alternates between horizontal and vertical movement
            if _ % 2 == 0:
                pos_x = (pos_x + speed_x) # % (canvas_size[1] - box_width)
            else:
                pos_y = (pos_y + speed_y) # % (canvas_size[0] - box_height)
        canvas[clamp(pos_y, 0, canvas_size[0]):clamp(pos_y + box_height, 0, canvas_size[0]),
                clamp(pos_x, 0, canvas_size[1]):clamp(pos_x + box_width, 0, canvas_size[1])] = 1

        # Add frame to the list
        frames.append(canvas)

    return frames


def generate_stationary_frames_simpler(canvas_size, num_frames, aspect_ratio, bounding_box_size):
    # Mapping size to bounding box dimensions
    size_mapping = {'Small': 0.25, 'Medium': 0.3, 'Large': 0.3}
    aspect_ratio_mapping = {'Rectangle Vertical': (1.33, 1), 'Rectangle Horizontal': (1, 1.33), 'Square': (1, 1)}

    # Calculate bounding box dimensions
    ratio = aspect_ratio_mapping[aspect_ratio]
    box_height = int(canvas_size[0] * size_mapping[bounding_box_size] * ratio[0])
    box_width = int(canvas_size[1] * size_mapping[bounding_box_size] * ratio[1])

    x_init_pos = [0.1 * canvas_size[1], 0.25 * canvas_size[1], 0.45*canvas_size[1], 0.7 * canvas_size[1]]
    y_init_pos = [0.1 * canvas_size[0], 0.25 * canvas_size[0], 0.45*canvas_size[0], 0.7 * canvas_size[0]]

    pos_x = np.random.choice(x_init_pos) + random.randint(int(-0.01 * canvas_size[1]), int(0.01 * canvas_size[1]))
    pos_y = np.random.choice(y_init_pos) + random.randint(int(-0.01 * canvas_size[0]), int(0.01 * canvas_size[0]))
    # Initialize frames
    frames = []
    for _ in range(num_frames):
        canvas = np.zeros(canvas_size)

        # Determine movement direction and apply movement
        pos_y = pos_y + random.randint(int(-0.01 * canvas_size[0]), int(0.01 * canvas_size[0]))
        pos_x = pos_x + random.randint(int(-0.01 * canvas_size[1]), int(0.01 * canvas_size[1]))

        canvas[clamp(pos_y, 0, canvas_size[0]):clamp(pos_y + box_height, 0, canvas_size[0]),
                clamp(pos_x, 0, canvas_size[1]):clamp(pos_x + box_width, 0, canvas_size[1])] = 1

        # Add frame to the list
        frames.append(canvas)


    return frames


input_file_path = "custom_prompts.csv"
output_file_path = "custom_prompts_with_bb.pkl"
num_videos_per_prompt = 3
video_id = 1100
all_records = []
frames_per_prompts = 3
num_frames = 16
with open('filtered_prompts.txt') as f:
    GOOD_PROMPTS = set([x.strip() for x in f.readlines()])
with open(input_file_path, "r") as f:
    data = csv.reader(f)
    for row in tqdm.tqdm(data):
        prompt = row[0]
        prompt = prompt.replace('herd of', '').replace('group of', '').replace('flock of', '').replace('school of', '').replace('escalator', 'elevator')
        subject = row[1].lower().replace('herd of', '').replace('group of', '').replace('flock of', '').replace('school of', '').replace('escalator', 'elevator')
        if prompt not in GOOD_PROMPTS:
            continue
        canvas_size = (224, 224)
        frames = []
        if row[-1] == "Stationary":
            for _ in range(frames_per_prompts):
                frames.append(generate_stationary_frames_simpler(canvas_size, num_frames, row[3], row[2]))
        else:
            for _ in range(frames_per_prompts):
                up_to_down_strict = False
                if "up" in prompt.lower() or 'ascending' in prompt.lower():
                    up_to_down_strict = 'up'
                elif "down" in prompt.lower() or 'descending' in prompt.lower():
                    up_to_down_strict = 'down'
                else:
                    up_to_down_strict = False
                frames.append(generate_moving_frames_simpler(canvas_size, num_frames, row[3], row[2], row[4], up_to_down_strict))

        for i in range(frames_per_prompts):
            record_dict = {"video_id": video_id, "prompt": prompt, "frames": frames[i], "subject": row[1], "motion": row[4], "aspect_ratio": row[3], "bounding_box_size": row[2]}
            all_records.append(record_dict)
            video_id += 1
    print(f"Wrote {len(all_records)} records to {output_file_path}")
    with open(output_file_path, "wb") as f:
        pickle.dump(all_records, f)


