import argparse
import json
import os
import random

import cv2
import habitat
from config_utils import r2r_train_config, rxr_train_config
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['r2r', 'rxr'], required=True, help='Dataset name')
    parser.add_argument('--part_idx', type=int, required=True, help='Part ID')
    parser.add_argument('--training_data_root_path', type=str, required=True, help='Training data root path')
    return parser.parse_args()


def build_config(dataset, part_idx):
    habitat_config = (r2r_train_config if dataset == 'r2r' else rxr_train_config)(stage='train', part_idx=str(part_idx), img_size=1024)
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = 120
    return habitat_config


def set_rgb_sensor_height(env, height):
    agent = env.sim.agents[0]
    agent_config = agent.agent_config
    rgb_sensor_spec = next(spec for spec in agent_config.sensor_specifications if spec.uuid in ['rgb', 'rgb_sensor'])
    rgb_sensor_spec.position = [0, height, 0]
    agent.reconfigure(agent_config, True)


def save_episode(env, path_planner, obs, traj_data_path, with_state):
    os.makedirs(traj_data_path, exist_ok=True)
    instruction = env.current_episode.instruction.instruction_text
    reference_path = env.current_episode.reference_path
    state = []
    step_idx = 0
    gt_act_sequences = []

    for reference_pos in reference_path[1:]:
        while True:
            act = path_planner.get_next_action(reference_pos)
            rgb = cv2.cvtColor(obs['rgb'], cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (384, 384))
            cv2.imwrite(os.path.join(traj_data_path, f"step_{step_idx}_rgb.jpg"), rgb)
            if with_state:
                state.append(env.sim.get_agent_state().sensor_states['rgb'].position)
            if act != 0:
                gt_act_sequences.append(act)
                obs = env.step(act)
                step_idx += 1
            else:
                break

    gt_act_sequences.append(0)
    env.step(0)
    with open(os.path.join(traj_data_path, "gt_acts.json"), 'w') as f:
        json.dump({"instruction": instruction, "gt_act_sequences": gt_act_sequences}, f, indent=4)
    if with_state:
        print('refer', reference_path, 'state', state)


if __name__ == "__main__":
    args = parse_args()
    habitat_config = build_config(args.dataset, args.part_idx)
    env = habitat.Env(habitat_config)
    path_planner = ShortestPathFollower(env.sim, 1, False)
    traj_num = 676 if args.dataset == 'r2r' else 3015

    for local_traj_idx in tqdm(range(traj_num)):
        obs = env.reset()
        if args.dataset == 'rxr' and env.current_episode.language not in ['en-US', 'en-IN']:
            continue
        set_rgb_sensor_height(env, random.uniform(0.5, 1.5))
        agent_state = env.sim.get_agent_state()
        obs = env.sim.get_observations_at(agent_state.position, agent_state.rotation, keep_agent_at_new_pose=True)
        traj_idx = traj_num * (args.part_idx - 1) + local_traj_idx if args.dataset == 'rxr' else local_traj_idx
        traj_name = f"traj_{traj_idx}_{env.current_episode.instruction.trajectory_id}" if args.dataset == 'r2r' else f"traj_{traj_idx}"
        save_episode(env, path_planner, obs, os.path.join(args.training_data_root_path, traj_name), args.dataset == 'r2r')
