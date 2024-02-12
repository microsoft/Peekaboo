import gradio as gr
import os
import cv2
import numpy as np
from PIL import Image
from moviepy.editor import *

def get_frames(video_in):
    frames = []
    #resize the video
    clip = VideoFileClip(video_in)
    
    #check fps
    if clip.fps > 30:
        print("vide rate is over 30, resetting to 30")
        clip_resized = clip.resize(height=512)
        clip_resized.write_videofile("video_resized.mp4", fps=30)
    else:
        print("video rate is OK")
        clip_resized = clip.resize(height=512)
        clip_resized.write_videofile("video_resized.mp4", fps=clip.fps)
    
    print("video resized to 512 height")
    
    # Opens the Video file with CV2
    cap= cv2.VideoCapture("video_resized.mp4")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("video fps: " + str(fps))
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite('kang'+str(i)+'.jpg',frame)
        frames.append('kang'+str(i)+'.jpg')
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()
    print("broke the video into frames")
    
    return frames, fps


def convert(gif):
    if gif != None:
        clip = VideoFileClip(gif.name)
        clip.write_videofile("my_gif_video.mp4")
        return "my_gif_video.mp4"
    else:
        pass


def create_video(frames, fps, type):
    print("building video result")
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(type + "_result.mp4", fps=fps)
    
    return type + "_result.mp4"