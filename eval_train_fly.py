from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import copy
import os
import torch
import warnings
import numpy as np
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from tqdm import tqdm
import habitat
import logging
from config_utils import *
import re
import cv2
import json
import argparse
warnings.filterwarnings("ignore")
import magnum as mn

def process_images_as_video(images, original_fps, max_frames_num, target_fps=1, force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 384, 384, 3)), "0.00s", 0.0

    total_frames = len(images)
    video_duration = total_frames / original_fps

    sampling_interval = max(1, round(original_fps / target_fps))
    frame_indices = list(range(0, total_frames, sampling_interval))
    time_stamps = [idx/original_fps for idx in frame_indices]

    if len(frame_indices) > max_frames_num or force_sample:
        uniform_samples = np.linspace(0, total_frames-1, max_frames_num, dtype=int)
        frame_indices = uniform_samples.tolist()
        time_stamps = [idx/original_fps for idx in frame_indices]

    time_str = ",".join(f"{t:.2f}s" for t in time_stamps)


    sampled_frames = np.stack([images[i] for i in frame_indices])

    return sampled_frames, time_str, video_duration


def llava_inference(conv_template, video, frame_time, video_time, tokenizer, instr, gt_step):

    multi_step_prompt = f""" You are navigating in an indoor environment given the instruction: {instr};
        You are given the observation history of previous steps you have taken;
        You should:
        1) evaluate the history to decide which step of instruction you are at.
        2) Predict actions for the next {gt_step} steps to follow up the given instruction until you reach the goal;
        Notice that:
        1) You can only choose from the following four actions: Move forward, Turn left, Turn right, Stop;
        2) Move forward means to move 0.25 meters straight ahead, and turning left or right is a 30-degree turn.
        3) If you believe you have reached the target or caught in obstacles, you should choose the stop action.
        ----
        Starting below, you should strictly follow this format:
        Final Answer: Your predicted actions for the next {gt_step} steps"""

    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
    question = DEFAULT_IMAGE_TOKEN + f"\n{time_instruciton}\n{multi_step_prompt}"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    with torch.amp.autocast(device_type="cuda"):
        cont = model.generate(
            input_ids,
            images=video,
            modalities= ["video"],
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()

    return text_outputs

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['r2r', 'rxr'], required=True, help='Dataset name')
    parser.add_argument('--id', type=int, required=True, help='Model ID')
    parser.add_argument('--pretrained', type=str, required=True, help='Pretrained model path')
    parser.add_argument('--training_data_root_path', type=str, required=True, help='Training data root path')
    parser.add_argument('--log_dir', type=str, required=True, help='Log directory')
    parser.add_argument('--ckpt_chosen', type=str, default=None, help='Checkpoint tag')
    return parser.parse_args()


def dataset_settings(dataset):
    return {
        'r2r': {
            'config': r2r_train_config,
            'parts': [[1,2],[4,5],[7,8],[9,10],[11,12],[13,14],[15,16],[3,6]],
            'traj_num': 676,
            'ckpt_chosen': 'aga2-r2r',
            'skip_existing': False,
            'draw_images': False,
            'filter_language': False,
        },
        'rxr': {
            'config': rxr_train_config,
            'parts': [[1,2,3],[4,5,6],[7,8,9],[10,11],[13,14],[15,16],[17,18],[12,19,20]],
            'traj_num': 3015,
            'ckpt_chosen': 'new-25',
            'skip_existing': True,
            'draw_images': True,
            'filter_language': True,
        },
    }[dataset]


def parse_action(text):
    match = re.search(r'\b(forward|turn left|turn right|stop)\b', text, re.IGNORECASE)
    if match:
        return {'forward': 1, 'turn left': 2, 'turn right': 3, 'stop': 0}[match.group(0).lower()]
    logging.warning("No Match Action. Repredict")
    print("No Match Action. Repredict")


def save_rgb(obs, traj_data_path, step_idx, suffix):
    rgb = cv2.cvtColor(obs['rgb'],cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (384,384))
    cv2.imwrite(os.path.join(traj_data_path, f"step_{step_idx}_{suffix}.jpg"), rgb)
    return rgb


def run_episode(env, path_planner, obs, traj_data_path, instr, instruction, part, gt_step, max_steps, draw_images):
    reference_path = env.current_episode.reference_path
    step_idx = 0
    save_step = []
    save_pose = []
    save = []
    flag = False
    rgb_input_traj = []

    while True:
        if step_idx == 0:
            rgb_input_traj.append(save_rgb(obs, traj_data_path, step_idx, 'rgb'))
            save_pose.append(copy.deepcopy(env.sim.get_agent_state().sensor_states['rgb'].position))
            save.append(copy.deepcopy(env.sim.get_agent_state()))
        video, frame_time, video_time = process_images_as_video(rgb_input_traj, original_fps=1, max_frames_num=8, target_fps=1, force_sample=False)
        video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
        output = llava_inference(conv_template, [video], frame_time, video_time, tokenizer, instr, gt_step).split(',')
        for i in range(4):
            act = parse_action(output[i])
            step_idx += 1
            if step_idx == max_steps:
                act = 0
            if act == 0:
                flag = True
                break
            obs = env.step(int(act))
            save_pose.append(copy.deepcopy(env.sim.get_agent_state().sensor_states['rgb'].position))
            save.append(copy.deepcopy(env.sim.get_agent_state()))
            save_step.append(act)
            rgb_input_traj.append(save_rgb(obs, traj_data_path, step_idx, 'rgb'))
        if flag:
            break

    refer = inter(reference_path,0.3)
    index = find_point(refer,save_pose,0.3+0.08*part)
    gt_act_sequences = []
    if index:
        agent = env.sim.agents[0]
        agent.set_state(save[index])
        min = find_min(refer,save_pose,index)
        obs = env.sim.get_observations_at(agent.state.position, agent.state.rotation, keep_agent_at_new_pose=True)
        state = []
        step_idx = 0
        indice = find_ori(reference_path,refer,min)
        if indice >= 0:
            for reference_pos in reference_path[indice:]:
                while True:
                    act = path_planner.get_next_action(mn.Vector3(reference_pos))
                    rgb = cv2.cvtColor(obs['rgb'],cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, (384,384))
                    cv2.imwrite(os.path.join(traj_data_path, f"step_{step_idx}_cor_rgb.jpg"), rgb)
                    state.append(env.sim.get_agent_state().sensor_states['rgb'].position)
                    if act != 0:
                        gt_act_sequences.append(act)
                        obs = env.step(act)
                        step_idx += 1
                    else:
                        break
        gt_act_sequences.append(0)
        env.step(0)
        if draw_images:
            if len(state) != 0:
                ima(refer,state,os.path.join(traj_data_path, '1.png'),0)
            ima(refer,save_pose,os.path.join(traj_data_path, '2.png'),index)

    with open(os.path.join(traj_data_path, "gt_acts.json"), 'w') as f:
        json.dump({"instruction": instruction, "gt_act_sequences": gt_act_sequences, "act": save_step, "point": index}, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    settings = dataset_settings(args.dataset)
    max_steps = 150
    ckpt_chosen = args.ckpt_chosen or settings['ckpt_chosen']
    pretrained = args.pretrained
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)
    model.eval()
    conv_template = "qwen_1_5"
    gt_step = 6
    logging.basicConfig(filename=os.path.join(args.log_dir, pretrained.split('/')[-2] + f"_ckpt-{ckpt_chosen}.log"), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    for part in settings['parts'][args.id]:
        habitat_config = settings['config'](stage='train',part_idx=str(part),img_size=1024)
        habitat_config.habitat.environment.max_episode_steps = 5000
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = 120
        env = habitat.Env(habitat_config)
        path_planner = ShortestPathFollower(env.sim,1,False)

        for traj_idx in tqdm(range(settings['traj_num'])):
            traj_idx = settings['traj_num'] * (part - 1) + traj_idx
            obs = env.reset()
            if settings['filter_language'] and env.current_episode.language not in ['en-US', 'en-IN']:
                continue
            traj_data_path = os.path.join(args.training_data_root_path, f"traj_{part}_{traj_idx}_{obs['instruction']['trajectory_id']}")
            if settings['skip_existing'] and os.path.exists(traj_data_path):
                continue
            if not os.path.exists(traj_data_path):
                os.makedirs(traj_data_path)
            run_episode(env, path_planner, obs, traj_data_path, obs['instruction']['text'], env.current_episode.instruction.instruction_text, part, gt_step, max_steps, settings['draw_images'])
