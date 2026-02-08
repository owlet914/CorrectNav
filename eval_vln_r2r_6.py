from CorrectNav.model.builder import load_pretrained_model
from CorrectNav.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from CorrectNav.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from CorrectNav.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import os

import torch
import sys
import warnings
from decord import VideoReader, cpu
import numpy as np
from filelock import FileLock

import habitat
import logging
from config_utils import *
import numpy as np
import base64
from io import BytesIO
import re
import time
import cv2
import json
import argparse
import textwrap
import math
from habitat.utils.visualizations.maps import colorize_draw_agent_and_fit_to_height
warnings.filterwarnings("ignore")


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
    
    # 提取帧数据
    sampled_frames = np.stack([images[i] for i in frame_indices])
    
    return sampled_frames, time_str, video_duration


def correct_inference(conv_template, video, frame_time, video_time, tokenizer, instr, gt_step):
    # time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {len(video[0])} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
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

def put_text_with_autowrap(img, text, position, font_scale=1, color=(0,0,0), thickness=2):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    lines = textwrap.wrap(text, width=int(img.shape[1] / (text_width / len(text))))
    
    y = position[1]
    for line in lines:
        textsize = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        gap = textsize[1] + 10
        y += gap
        cv2.putText(img, line, (position[0], y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    
    return img
def locked_load_json(json_path):
    lock_path = json_path + ".lock"
    with FileLock(lock_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data

def locked_dump_json(data, json_path):
    lock_path = json_path + ".lock"
    with FileLock(lock_path):
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, required=True, help='Model ID')
    
    args = parser.parse_args()   
    
    idt=args.model_id 
    ckpt_chosen = "sota-r2r"   
    pretrained="YOUR MODEL PATH"
    model_name = "correctnav"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, torch_dtype="bfloat16", device_map=device_map)  # Add any other thing you want to pass in llava_model_args
    model.eval()
    conv_template = "qwen_1_5"  
    log_file = pretrained.split('/')[-2] + f"/_ckpt-{ckpt_chosen}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    habitat_config = r2r_w_config(stage='val_unseen',img_size=1024)
    habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov=120
    env = habitat.Env(habitat_config)  
    traj_ids = []
    success_rates, spl_results, dtg_results = [], [], []
    inf_time = []
    num = 0
    max_steps = 162
    gt_step = 6
    count=0
    while True:
        obs = env.reset()
        instr = obs['instruction']['text']
        file_path = os.path.join('/LLaVA-NeXT/vln_eval', pretrained.split('/')[-2] + f"_ckpt-{ckpt_chosen}.json")
        if os.path.exists(file_path):
             file=locked_load_json(file_path)                
        else:
            file = []
        instructions = [list(item.keys())[0] for item in file]
        if instr in instructions:
            continue       
        data = {instr:(0,0,0,0)}
        file.append(data)
        locked_dump_json(file,file_path)        
        num = num + 1
        logging.info(f"Instruction: {instr}")
        combined_traj=[]
        rgb_input_traj = []
        rgb_video_traj = []
        topdown_traj = []
        t_pertraj = 0
        positions = [] 
        flag = False
        traj_id = obs['instruction']['trajectory_id']     
        f=True
        step_idx=0        
        Os=0
        while True:  
            if step_idx==0:
                rgb = cv2.cvtColor(obs['rgb'],cv2.COLOR_BGR2RGB)
                rgb=cv2.resize(rgb, (384,384))
                rgb_input_traj.append(rgb)
            video, frame_time, video_time = process_images_as_video(rgb_input_traj, original_fps = 1, max_frames_num = 12, target_fps=1, force_sample=False)
            video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
            video = [video]
            output = correct_inference(conv_template, video, frame_time, video_time,  tokenizer, instr2, gt_step)
            output=output.split(',')
            for i in range(4):               
                match = re.search(r'\b(forward|turn left|turn right|stop)\b', output[i], re.IGNORECASE)
                if match:
                    action = match.group(0).lower()  # 获取匹配的动作并转换为小写
                    if action == 'forward':
                        act = 1
                    elif action == 'turn left':
                        act = 2
                    elif action == 'turn right':
                        act = 3
                    elif action == 'stop':
                        act = 0
                    else:
                        logging.warning('no action in output!')
                else:
                    logging.warning("No Match Action. Repredict")
                    print("No Match Action. Repredict")              
                step_idx+=1
                if step_idx == max_steps :
                    act = 0
                obs = env.step(int(act))
                metrics = env.get_metrics()
                if metrics["distance_to_goal"]<3:
                    Os=1              
                if env.episode_over or flag:                
                    break
                rgb = cv2.cvtColor(obs['rgb'],cv2.COLOR_BGR2RGB)
                rgb=cv2.resize(rgb, (384,384))
                rgb_input_traj.append(rgb)       
            if env.episode_over or flag:             
                break                            
        metrics = env.get_metrics()
        print(list(metrics.keys()))
        success, spl, dtg= metrics['success'], metrics['spl'], metrics["distance_to_goal"]
        print(instr, success, spl, dtg)

        file=locked_load_json(file_path)
        for i, item in enumerate(file):
            if instr == list(item.keys())[0]:
                file[i][instr] = (success, spl, dtg, Os)
                break  
        locked_dump_json(file,file_path)

        success_rates, spl_results, dtg_results,os_results = [], [], [],[]
        for item in file:
            result = list(item.values())[0]
            if all(math.isfinite(x) for x in result):
                success_rates.append(result[0])
                spl_results.append(result[1])
                dtg_results.append(result[2])
                os_results.append(result[3])


        avg_sr, avg_spl, avg_dtg ,avg_os= np.mean(success_rates), np.mean(spl_results), np.mean(dtg_results), np.mean(os_results)
        
        num_eval = len(success_rates)
        eval_result = 'Success:%.3f,SPL:%.2f,Nav Error:%.2f,OS:%.3f'%(avg_sr, avg_spl, avg_dtg,avg_os)
        logging.info(f"[{num_eval}] Eval Results: "+eval_result)

