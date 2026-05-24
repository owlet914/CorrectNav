import argparse
import glob
import json
import os

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_training_data_path', type=str, required=True, help='Raw training data path')
    parser.add_argument('--video_root_path', type=str, required=True, help='Video root path written to JSON')
    parser.add_argument('--output_json_path', type=str, required=True, help='Output JSON path')
    parser.add_argument('--data_source', type=str, required=True, help='Data source name')
    return parser.parse_args()


def traj_sort_key(traj_dir):
    return tuple(int(item) for item in traj_dir.split('_')[1:] if item.isdigit())


if __name__ == "__main__":
    args = parse_args()
    data = []

    for num_id, traj_dir in enumerate(tqdm(sorted(os.listdir(args.raw_training_data_path), key=traj_sort_key))):
        with open(os.path.join(args.raw_training_data_path, traj_dir, "gt_acts.json"), 'r') as file:
            gt_acts_dict = json.load(file)

        video_output_path = os.path.join(args.video_root_path, traj_dir, f"step_{len(glob.glob(os.path.join(args.raw_training_data_path, traj_dir, '*.jpg'))) - 1}_video.mp4")
        if not os.path.exists(video_output_path):
            raise FileNotFoundError(video_output_path)

        multi_step_prompt = """ You are navigating an indoor environment based on human instructions.
    Your observation is a first-person video stream captured during movement.
    Analyze the visual sequence carefully and infer the complete original instruction that guided this navigation path"""
        data.append({
            "id": str(num_id),
            "conversations": [
                {"from": "human", "value": f"<image>\n{multi_step_prompt}"},
                {"from": "gpt", "value": gt_acts_dict["instruction"]},
            ],
            "data_source": args.data_source,
            "video": video_output_path,
        })

    with open(args.output_json_path, 'w') as f:
        json.dump(data, f, indent=4)
