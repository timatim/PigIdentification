"""
Extracts images from video using OpenCV
"""
import cv2
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--train-dir', type=str, default='./train')
parser.add_argument('--output-dir', type=str, default='./train/frames')
args = parser.parse_args()

train_dir = args.train_dir
output_dir = args.output_dir

mp4s = [f for f in os.listdir(train_dir) if f.split('.')[-1] == 'mp4']

for f in tqdm(mp4s):
    frames_dir = os.path.join(output_dir, f.split('.')[0])
    
    # make directory
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    # read frames
    vidcap = cv2.VideoCapture(os.path.join(train_dir, f))
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        cv2.imwrite(os.path.join(frames_dir, "frame%d.jpg" % count), image)
        count += 1