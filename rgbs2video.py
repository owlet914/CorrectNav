import argparse
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


if __name__ == "__main__":
    args = parse_args()
    traj_dir_list = sorted(os.listdir(args.raw_training_data_path), key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))

    for traj_dir in tqdm(get_sublist(traj_dir_list, args.part_idx, args.n_part)):
        if not os.path.exists(os.path.join(args.raw_training_data_path, traj_dir, "1.png")):
            continue
        if not os.path.exists(os.path.join(args.raw_training_data_path, traj_dir, "gt_acts.json")):
            continue

        with open(os.path.join(args.raw_training_data_path, traj_dir, "gt_acts.json"), 'r') as file:
            gt_acts_dict = json.load(file)
        if gt_acts_dict['gt_act_sequences'] == []:
            continue

        point = gt_acts_dict['point']
        gt_act_sequences = gt_acts_dict["gt_act_sequences"]
        name_list = []
        for i in range(point):
            name_list.append(os.path.join(args.raw_training_data_path, traj_dir, f"step_{i}_rgb.jpg"))
        for i in range(len(gt_act_sequences)):
            name_list.append(os.path.join(args.raw_training_data_path, traj_dir, f"step_{i}_cor_rgb.jpg"))

        target_dir = os.path.join(args.target_training_data_path, traj_dir)
        if os.path.exists(target_dir):
            continue
        os.makedirs(target_dir)
        img_sequence = []
        for step_idx in range(len(name_list)):
            img = cv2.imread(name_list[step_idx])
            img_sequence.append(img)
            if step_idx < point:
                continue
            video_output_path = os.path.join(target_dir, f"step_{step_idx}_video.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width, layers = img.shape
            video = cv2.VideoWriter(video_output_path, fourcc, 1.0, (width, height))
            for img in img_sequence:
                video.write(img)
            video.release()
