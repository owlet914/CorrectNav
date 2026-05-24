import argparse
import glob
import json
import os

import cv2
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part_idx', type=int, required=True, help='Part ID')
    parser.add_argument('--n_part', type=int, required=True, help='Number of parts')
    parser.add_argument('--raw_training_data_path', type=str, required=True, help='Raw training data path')
    parser.add_argument('--target_training_data_path', type=str, required=True, help='Target training data path')
    return parser.parse_args()


def get_sublist(full_list, idx, n_parts):
    chunk_size = len(full_list) // n_parts
    return full_list[(idx - 1) * chunk_size:idx * chunk_size]


def traj_sort_key(traj_dir):
    return tuple(int(item) for item in traj_dir.split('_')[1:] if item.isdigit())


if __name__ == "__main__":
    args = parse_args()
    traj_dir_list = sorted(os.listdir(args.raw_training_data_path), key=traj_sort_key)

    for traj_dir in tqdm(get_sublist(traj_dir_list, args.part_idx, args.n_part)):
        with open(os.path.join(args.raw_training_data_path, traj_dir, "gt_acts.json"), 'r') as file:
            json.load(file)

        target_dir = os.path.join(args.target_training_data_path, traj_dir)
        os.makedirs(target_dir, exist_ok=True)
        img_sequence = []
        for step_idx in range(len(glob.glob(os.path.join(args.raw_training_data_path, traj_dir, '*.jpg')))):
            img = cv2.imread(os.path.join(args.raw_training_data_path, traj_dir, f"step_{step_idx}_rgb.jpg"))
            img_sequence.append(img)
            video_output_path = os.path.join(target_dir, f"step_{step_idx}_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width, layers = img.shape
            video = cv2.VideoWriter(video_output_path, fourcc, 1.0, (width, height))
            for img in img_sequence:
                video.write(img)
            video.release()
