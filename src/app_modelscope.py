from models.pipelines import TextToVideoSDPipelineSpatialAware
import torch.nn.functional as F
import torch
import cv2
import sys
import gradio as gr
import os
import numpy as np
from gradio_utils import *


def image_mod(image):
    return image.rotate(45)


sys.path.insert(1, os.path.join(sys.path[0], '..'))


NUM_POINTS = 3
NUM_FRAMES = 16
LARGE_BOX_SIZE = 176


def generate_video(pipe, overall_prompt, latents, get_latents=False, num_frames=24, num_inference_steps=50, fg_masks=None,
                   fg_masked_latents=None, frozen_steps=0, custom_attention_mask=None, fg_prompt=None):

    video_frames = pipe(overall_prompt, num_frames=num_frames, latents=latents, num_inference_steps=num_inference_steps, frozen_mask=fg_masks,
                        frozen_steps=frozen_steps, latents_all_input=fg_masked_latents, custom_attention_mask=custom_attention_mask, fg_prompt=fg_prompt,
                        make_attention_mask_2d=True, attention_mask_block_diagonal=True, height=256, width=256).frames
    if get_latents:
        video_latents = pipe(overall_prompt, num_frames=num_frames, latents=latents,
                             num_inference_steps=num_inference_steps, output_type="latent").frames
        return video_frames, video_latents

    return video_frames


def interpolate_points(points, target_length):
    # print(points)
    if len(points) == target_length:
        return points
    elif len(points) > target_length:
        # Subsample the points uniformly
        indices = np.round(np.linspace(
            0, len(points) - 1, target_length)).astype(int)
        return [points[i] for i in indices]
    else:
        # Linearly interpolate to get more points
        interpolated_points = []
        num_points_to_add = target_length - len(points)
        points_added_per_segment = num_points_to_add // (len(points) - 1)

        for i in range(len(points) - 1):
            start, end = points[i], points[i + 1]
            interpolated_points.append(start)
            for j in range(1, points_added_per_segment + 1):
                fraction = j / (points_added_per_segment + 1)
                new_point = np.round(start + fraction * (end - start))
                interpolated_points.append(new_point)

        # Add the last point
        interpolated_points.append(points[-1])

        # If there are still not enough points, add extras at the end
        while len(interpolated_points) < target_length:
            interpolated_points.append(points[-1])

        return interpolated_points


torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


try:
    pipe = TextToVideoSDPipelineSpatialAware.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float, variant="fp32").to(torch_device)
except:
    pipe = TextToVideoSDPipelineSpatialAware.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float, variant="fp32").to(torch_device)


def generate_bb(prompt, fg_object, aspect_ratio, size, motion_direction, seed, peekaboo_steps, trajectory):

    if not set(fg_object.split()).issubset(set(prompt.split())):
        raise gr.Error("Foreground object should be present in the video prompt")
    # if len(trajectory['layers']) < NUM_POINTS:
    #   raise ValueError
    final_canvas = torch.zeros((NUM_FRAMES, 256//8, 256//8))

    bbox_size_x = LARGE_BOX_SIZE if size == "large" else int(
        LARGE_BOX_SIZE * 0.75) if size == "medium" else LARGE_BOX_SIZE//2
    bbox_size_y = bbox_size_x if aspect_ratio == "square" else int(
        bbox_size_x * 1.33) if aspect_ratio == "horizontal" else int(bbox_size_x * 0.75)

    bbox_coords = []

    image = trajectory['composite']
    # print(image.shape)

    image = cv2.resize(image, (256, 256))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour
    bbox_points = []
    for contour in contours:
        # You can approximate the contour to reduce the number of points
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Extracting and printing coordinates
        for point in approx:
            y, x = point.ravel()
            if x in range(1, 255) and y in range(1, 255):
                #   bbox_points.append([min(max(x, 32), 256-32),min(max(y, 32), 256-32)])
                bbox_points.append([min(max(x, 0), 256), min(max(y, 0), 256)])

    if motion_direction in ['Left to Right', 'Right to Left']:
        sorted_points = sorted(
            bbox_points, key=lambda x: x[1], reverse=motion_direction == "Right to Left")
    else:
        sorted_points = sorted(
            bbox_points, key=lambda x: x[0], reverse=motion_direction == "Down to Up")
    target_length = NUM_FRAMES
    final_points = interpolate_points(np.array(sorted_points), target_length)

    # Remember to reverse the co-ordinates
    for i in range(NUM_FRAMES):
        x = int(final_points[i][0])
        y = int(final_points[i][1])
        # Added Padding
        final_canvas[i, max(int(x-bbox_size_x/2), 0) // 8:min(int(x+bbox_size_x/2), 256) // 8,
                     max(int(y-bbox_size_y/2), 0) // 8:min(int(y+bbox_size_y/2), 256) // 8] = 1

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fg_masks = final_canvas.unsqueeze(1).to(torch_device)
#     # Save fg_masks as images
    for i in range(NUM_FRAMES):
        cv2.imwrite(f"./fg_masks/frame_{i:04d}.png",
                    fg_masks[i, 0].cpu().numpy()*255)

    seed = seed
    random_latents = torch.randn([1, 4, NUM_FRAMES, 32, 32], generator=torch.Generator(
    ).manual_seed(seed)).to(torch_device)
    overall_prompt = f"{prompt} , high quality"
    video_frames = generate_video(pipe, overall_prompt, random_latents, get_latents=False, num_frames=NUM_FRAMES, num_inference_steps=40,
                                  fg_masks=fg_masks, fg_masked_latents=None, frozen_steps=int(peekaboo_steps), fg_prompt=fg_object)
    video_frames_original = generate_video(pipe, overall_prompt, random_latents, get_latents=False, num_frames=NUM_FRAMES, num_inference_steps=40,
                                           fg_masks=None, fg_masked_latents=None, frozen_steps=0, fg_prompt=None)

    return create_video(video_frames_original, fps=8, type="modelscope"), create_video(video_frames, fps=8, type="final")


instructions_md = """
## Usage Instructions
- **Video Prompt**: Enter a brief description of the scene you want to generate.
- **Foreground Object**: Specify the main object in the video.
- **Aspect Ratio**: Choose the aspect ratio for the bounding box.
- **Size of the Bounding Box**: Select how large the foreground object should be.
- **Trajectory of the Bounding Box**: Draw the trajectory of the bounding box.
- **Motion Direction**: Indicate the direction of movement for the object.
- **Geek Settings**: Advanced settings for fine-tuning (optional).
- **Generate Video**: Click the button to create your video.

Feel free to experiment with different settings to see how they affect the output!
"""

with gr.Blocks() as demo:
    gr.Markdown("""
                # Peekaboo Demo
                """)
    with gr.Row():
        video_1 = gr.Video(label="Original Modelscope Video")
        video_2 = gr.Video(label="Peekaboo Video")
    

    with gr.Accordion(label="Usage Instructions", open=False):
        gr.Markdown(instructions_md)
    with gr.Group("User Input"):
        txt_1 = gr.Textbox(lines=1, label="Video Prompt", value="A panda walking in the forest.")
        txt_2 = gr.Textbox(lines=1, label="Foreground Object in the Video Prompt", value="panda")
        aspect_ratio = gr.Radio(choices=["square", "horizontal", "vertical"], label="Aspect Ratio", value="horizontal")
        trajectory = gr.Paint(value={'background': np.zeros((256, 256)), 'layers': [], 'composite': np.zeros((256, 256))}, type="numpy", image_mode="RGB", height=256, width=256, label="Trajectory of the Bounding Box")
        size = gr.Radio(choices=["small", "medium", "large"], label="Size of the Bounding Box", value="medium")
        motion_direction = gr.Radio(choices=["Left to Right", "Right to Left", "Up to Down", "Down to Up"], label="Motion Direction", value="Left to Right")

    with gr.Accordion(label="Geek settings", open=False):
        with gr.Group():
            seed = gr.Slider(0, 10, step=1., value=2, label="Seed")
            peekaboo_steps = gr.Slider(0, 20, step=1., value=2, label="Number of Peekaboo Steps")


    btn = gr.Button(value="Generate Video")
    
    btn.click(generate_bb, inputs=[txt_1, txt_2, aspect_ratio, size, motion_direction, seed, peekaboo_steps, trajectory], outputs=[video_1, video_2])




if __name__ == "__main__":
    demo.launch(share=True)
