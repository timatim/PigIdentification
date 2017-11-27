"""
Splits validation dataset from train dataset
"""
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train-dir', type=str, default='./train/frames', help='path to embedding')
parser.add_argument('--train-size', type=float, default=0.9, help='train ratio')
parser.add_argument('--valid-dir', type=str, default='./validation/frames', help='path to output validation data')

args = parser.parse_args()


train_dir = args.train_dir

# for each label, randomly select (1-train_size)*n and put into validation directory
for label in os.listdir(train_dir):
    if not os.path.isdir(os.path.join(train_dir, label)):
        continue
    frames = os.listdir(os.path.join(train_dir, label))
    valid_size = int((1 - args.train_size) * len(frames))
    # select
    valid_frames = np.random.choice(frames, valid_size, replace=False)

    # make destination dir if doesnt exist
    dest_dir = os.path.join(args.valid_dir, label)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # move files
    for valid_frame in valid_frames:
        os.rename(os.path.join(train_dir, label, valid_frame), os.path.join(dest_dir, valid_frame))

    print("Label %s, Train size %d, Validation size %d" %
          (label, len(os.listdir(os.path.join(train_dir, label))), len(os.listdir(dest_dir))))
