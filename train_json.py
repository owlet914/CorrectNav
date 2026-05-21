import argparse
import json
import os

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_training_data_path', type=str, required=True, help='Raw training data path')
    parser.add_argument('--video_root_path', type=str, required=True, help='Video root path written to JSON')
    parser.add_argument('--output_json_path', type=str, required=True, help='Output JSON path')
    parser.add_argument('--data_source', type=str, required=True, help='Data source name')
    parser.add_argument('--gt_step', type=int, default=6, help='Number of future actions')
    return parser.parse_args()


def multi_step_gt(gt_act_sequences, step_idx, gt_step):
    gt_act_lang = []
    for i in range(gt_step):
        if int(step_idx) + i > len(gt_act_sequences) - 1:
            break
        act = gt_act_sequences[int(step_idx) + i]
        if act == 1:
            gt_act_lang.append('Move forward')
        elif act == 2:
            gt_act_lang.append('Turn left')
        elif act == 3:
            gt_act_lang.append('Turn right')
        elif act == 0:
            gt_act_lang.append('Stop')
    while len(gt_act_lang) < gt_step:
        gt_act_lang.append('Stop')
    return ','.join(gt_act_lang)


if __name__ == "__main__":
    args = parse_args()
    data = []
    num_id = 0

    for traj_dir in tqdm(sorted(os.listdir(args.raw_training_data_path), key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2])))):
        if not os.path.exists(os.path.join(args.raw_training_data_path, traj_dir, "1.png")):
            continue
        if not os.path.exists(os.path.join(args.raw_training_data_path, traj_dir, "gt_acts.json")):
            continue

        with open(os.path.join(args.raw_training_data_path, traj_dir, "gt_acts.json"), 'r') as file:
            gt_acts_dict = json.load(file)
        if gt_acts_dict['gt_act_sequences'] == []:
            continue

        point = gt_acts_dict['point']
        instruction = gt_acts_dict["instruction"]
        gt_act_sequences = gt_acts_dict['act'][:point] + gt_acts_dict["gt_act_sequences"]
        for step_idx in range(len(gt_act_sequences)):
            num_id += 1
            if step_idx < point:
                continue
            video_output_path = os.path.join(args.video_root_path, traj_dir, f"step_{step_idx}_video.mp4")
            if not os.path.exists(video_output_path):
                raise FileNotFoundError(video_output_path)
            multi_step_prompt = f""" You are navigating in an indoor environment given the instruction: {instruction};
                You are given the observation history of previous steps you have taken;
                You should:
                1) evaluate the history to decide which step of instruction you are at.
                2) Predict actions for the next {args.gt_step} steps to follow up the given instruction until you reach the goal;
                Notice that:
                1) You can only choose from the following four actions: Move forward, Turn left, Turn right, Stop;
                2) Move forward means to move 0.25 meters straight ahead, and turning left or right is a 30-degree turn.
                3) If you believe you have reached the target or caught in obstacles, you should choose the stop action.
                ----
                Starting below, you should strictly follow this format:
                Final Answer: Your predicted actions for the next {args.gt_step} steps"""
            data.append({
                "id": str(num_id),
                "conversations": [
                    {"from": "human", "value": f"<image>\n{multi_step_prompt}"},
                    {"from": "gpt", "value": f"Final Answer: {multi_step_gt(gt_act_sequences, step_idx, args.gt_step)}"},
                ],
                "data_source": args.data_source,
                "video": video_output_path,
            })

    with open(args.output_json_path, 'w') as f:
        json.dump(data, f, indent=4)
