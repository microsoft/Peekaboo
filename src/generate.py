import os

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import warnings

import cv2
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
import torchvision.io as vision_io

from models.pipelines import TextToVideoSDPipelineSpatialAware
from diffusers.utils import export_to_video
from PIL import Image
import torchvision

import argparse

import warnings
warnings.filterwarnings("ignore")

def get_parser():
    parser = argparse.ArgumentParser(description="Generate videos with different prompts and fg objects")
    parser.add_argument("--model", type=str, default="zeroscope", choices=["zeroscope", "modelscope"], help="Model to use for the generation")
    parser.add_argument("--prompt", type=str, default="A panda eating bamboo in a lush bamboo forest.", help="Prompt to generate the video")
    parser.add_argument("--fg_object", type=str, default="panda", help="Foreground object to generate the video")
    parser.add_argument("--frozen_steps", type=int, default=2, help="Number of frozen steps")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=2, help="Seed for random number generation")
    parser.add_argument("--output_path", type=str, default="src/demo", help="Path to save the generated videos")
    return parser

def generate_video(pipe, overall_prompt, latents, get_latents=False, num_frames=24, num_inference_steps=50, fg_masks=None, 
        fg_masked_latents=None, frozen_steps=0, custom_attention_mask=None, fg_prompt=None, height=320, width=576):
    
    video_frames = pipe(overall_prompt, num_frames=num_frames, latents=latents, num_inference_steps=num_inference_steps, frozen_mask=fg_masks, 
    frozen_steps=frozen_steps, latents_all_input=fg_masked_latents, custom_attention_mask=custom_attention_mask, fg_prompt=fg_prompt,
    make_attention_mask_2d=True, attention_mask_block_diagonal=True, height=height, width=width ).frames
    if get_latents:
        video_latents = pipe(overall_prompt, num_frames=num_frames, latents=latents, num_inference_steps=num_inference_steps, output_type="latent").frames
        return video_frames, video_latents
    
    return video_frames

def save_frames(path):
    video, audio, video_info = vision_io.read_video(f"{path}.mp4", pts_unit='sec')

    # Number of frames
    num_frames = video.size(0)

    # Save each frame
    os.makedirs(f"{path}", exist_ok=True)
    for i in range(num_frames):
        frame = video[i, :, :, :].numpy()
        # Convert from C x H x W to H x W x C and from torch tensor to PIL Image
        # frame = frame.permute(1, 2, 0).numpy()
        img = Image.fromarray(frame.astype('uint8'))
        img.save(f"{path}/frame_{i:04d}.png")

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    assert args.frozen_steps <= args.num_inference_steps, "Frozen steps should be less than or equal to the number of inference steps"
    assert args.fg_object in args.prompt, "Foreground object should be present in the prompt"
    # Example usage
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate video

    if args.model == "zeroscope":
        pipe = TextToVideoSDPipelineSpatialAware.from_pretrained(
            "cerspense/zeroscope_v2_576w", torch_dtype=torch.float, variant="fp32").to(torch_device)
        num_frames = 24
        random_latents = torch.randn([1, 4, num_frames, 40, 72], generator=torch.Generator().manual_seed(args.seed)).to(torch_device)
        bbox_mask = torch.zeros([num_frames, 1, 40, 72], device=torch_device)
        height = 320
        width = 576
    elif args.model == "modelscope":
        pipe = TextToVideoSDPipelineSpatialAware.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float, variant="fp32").to(torch_device)
        num_frames=16
        random_latents = torch.randn([1, 4, num_frames, 32, 32], generator=torch.Generator().manual_seed(args.seed)).to(torch_device)
        bbox_mask = torch.zeros([num_frames, 1, 32, 32], device=torch_device)
        height = 256
        width = 256

    # Simulate a moving bounding box
    x_start = [10 + (i % 3) for i in range(num_frames)]  # Simulating slight movement in x
    x_end = [30 + (i % 3) for i in range(num_frames)]    # Simulating slight movement in x
    y_start = [10 for _ in range(num_frames)]            # Static y start 
    y_end = [25 for _ in range(num_frames)]              # Static y end

    # Populate the bbox_mask tensor with ones where the bounding box is located
    for i in range(num_frames):
        bbox_mask[i, :, x_start[i]:x_end[i], y_start[i]:y_end[i]] = 1

    fg_masks = bbox_mask
    
    fg_masked_latents = None
    prompts = [(args.fg_object, args.prompt)]

    save_path = args.model
    for fg_object, overall_prompt in prompts:
        os.makedirs(f"{args.output_path}/{save_path}/{overall_prompt}-mask", exist_ok=True)
        for i in range(num_frames):
            torchvision.utils.save_image(fg_masks[i], f"{args.output_path}/{save_path}/{overall_prompt}-mask/frame_{i:04d}.png")

        print("Generating video for prompt: ", overall_prompt)
        for frozen_steps in [args.frozen_steps]: # try different frozen steps

            video_frames = generate_video(pipe, overall_prompt, random_latents, get_latents=False, num_frames=num_frames, num_inference_steps=args.num_inference_steps, 
                fg_masks=fg_masks, fg_masked_latents=fg_masked_latents, frozen_steps=frozen_steps, fg_prompt=fg_object, height=height, width=width)
            
            # Save video frames
            overall_prompt = overall_prompt.replace(" ", "_")
            os.makedirs(f"{args.output_path}/{save_path}/{overall_prompt}", exist_ok=True)
            video_path = export_to_video(video_frames, f"{args.output_path}/{save_path}/{overall_prompt}/{frozen_steps}_of_{args.num_inference_steps}_{args.seed}_peekaboo.mp4")
            save_frames(f"{args.output_path}/{save_path}/{overall_prompt}/{frozen_steps}_of_{args.num_inference_steps}_{args.seed}_peekaboo")
            print(f"Video saved at {video_path}")

