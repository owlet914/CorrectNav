import habitat
import os
import cv2
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    FogOfWarConfig,
    TopDownMapMeasurementConfig,
    LookUpActionConfig,
    LookDownActionConfig
)
import matplotlib.pyplot as plt
import numpy as np
HM3D_CONFIG_PATH = "/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
MP3D_CONFIG_PATH = "/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_mp3d.yaml"
PROCTHOR_CONFIG_PATH = "/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_procthor-hab.yaml"
HSSD_CONFIG_PATH = "/habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hssd-hab.yaml"
R2R_CONFIG_PATH = "/habitat-lab/habitat-lab/habitat/config/benchmark/nav/vln_r2r.yaml"   
RXR_CONFIG_PATH = "/habitat-lab/habitat-lab/habitat/config/benchmark/nav/vln_rxr.yaml"    

def hm3d_config(path:str=HM3D_CONFIG_PATH,stage:str='val'):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = "/habitat-data-0.2.5/scenes"
        habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/objectnav/hm3d/v2/{split}/{split}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = "/habitat-data-0.2.5/scenes/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25
    return habitat_config
    
def hssd_config(path:str=HSSD_CONFIG_PATH,stage:str='val'):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = "/habitat-data-0.2.5/scenes/hssd-hab/scenes"
        habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/objectnav/hssd-hab/{split}/{split}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = "/habitat-data-0.2.5/scenes/hssd-hab/hssd-hab.scene_dataset_config.json"
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
    return habitat_config
    
def mp3d_config(path:str=MP3D_CONFIG_PATH,stage:str='val'):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = "/habitat-data-0.2.5/scenes"
        habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/objectnav/mp3d/v1/{split}/{split}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = "/habitat-data-0.2.5/scenes/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 0.25
    return habitat_config

def procthor_config(path:str=PROCTHOR_CONFIG_PATH,stage:str='val'):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = "/habitat-data-0.2.5/scenes/ai2thor-hab/ai2thor-hab/configs/scenes/ProcTHOR"
        habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/objectnav/procthor-hab/{split}/{split}.json.gz"
        habitat_config.habitat.simulator.scene_dataset = "/habitat-data-0.2.5/scenes/ai2thor-hab/ai2thor-hab/ai2thor-hab.scene_dataset_config.json"
        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
    return habitat_config

def r2r_config(path:str=R2R_CONFIG_PATH,stage:str='val_seen', part_idx=None, img_size:int=384):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = "/habitat-data-0.2.5/scenes"
        # habitat_config.habitat.dataset.data_path = "/vepfs-cnsh4137610c2f4c/algo/user9/habitat-data-0.2.5/datasets/vlnnav/r2r/{split}/{split}.json.gz"
        if part_idx == None:
            habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/vlnnav/r2r/{split}/sampled613_{split}.json.gz"
        else:
            habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/vlnnav/r2r/{split}/sampled613_{split}_partidx.json.gz"
            habitat_config.habitat.dataset.data_path = habitat_config.habitat.dataset.data_path.replace("idx", part_idx)
        habitat_config.habitat.simulator.scene_dataset = "/habitat-data-0.2.5/scenes/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.simulator.turn_angle = 30

        habitat_config.habitat.task.actions.update({
            "look_up": LookUpActionConfig(tilt_angle=30),
            "look_down": LookDownActionConfig(tilt_angle=30)
        })

        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig()            
        })

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov=90
        
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov=90
        
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 3.0
        
    return habitat_config
def r2r_w_config(path:str=R2R_CONFIG_PATH,stage:str='val_unseen', part_idx=None, img_size:int=384):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = "/habitat-data-0.2.5/scenes"
        # habitat_config.habitat.dataset.data_path = "/vepfs-cnsh4137610c2f4c/algo/user9/habitat-data-0.2.5/datasets/vlnnav/r2r/{split}/{split}.json.gz"
        if part_idx == None:
            habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/vlnnav/r2r/{split}/{split}.json.gz"
        else:
            habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/vlnnav/r2r/{split}/{split}_partidx.json.gz"
            habitat_config.habitat.dataset.data_path = habitat_config.habitat.dataset.data_path.replace("idx", part_idx)
        habitat_config.habitat.simulator.scene_dataset = "/habitat-data-0.2.5/scenes/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.simulator.turn_angle = 30

        habitat_config.habitat.task.actions.update({
            "look_up": LookUpActionConfig(tilt_angle=30),
            "look_down": LookDownActionConfig(tilt_angle=30)
        })

        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov=90
        
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov=90
        
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 3.0
        
    return habitat_config

def r2r_train_config(path:str=R2R_CONFIG_PATH,stage:str='val_seen', part_idx=None, img_size:int=384):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = "/habitat-data-0.2.5/scenes"
        if part_idx == None:
            habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/vlnnav/r2r/{split}/{split}.json.gz"
        else:
            habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/vlnnav/r2r/{split}/{split}_partidx.json.gz"
            habitat_config.habitat.dataset.data_path = habitat_config.habitat.dataset.data_path.replace("idx", part_idx)
        print(habitat_config.habitat.dataset.data_path)
        habitat_config.habitat.simulator.scene_dataset = "/habitat-data-0.2.5/scenes/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.simulator.turn_angle = 30

        habitat_config.habitat.task.actions.update({
            "look_up": LookUpActionConfig(tilt_angle=30),
            "look_down": LookDownActionConfig(tilt_angle=30)
        })

        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov=90
        
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov=90
        
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 3.0
        
    return habitat_config

def rxr_train_config(path:str=RXR_CONFIG_PATH,stage:str='train',part_idx=None,img_size:int=384):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = "/habitat-data-0.2.5/scenes"
        if part_idx == None:
            habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/vlnnav/rxr/{split}/{split}.json.gz"
        else:
            habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/vlnnav/rxr/{split}/{split}_partidx.json.gz"
            habitat_config.habitat.dataset.data_path = habitat_config.habitat.dataset.data_path.replace("idx", part_idx)
        print(habitat_config.habitat.dataset.data_path)
        #habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/vlnnav/rxr/{split}/{split}_guide.json.gz"
        habitat_config.habitat.simulator.scene_dataset = "/habitat-data-0.2.5/scenes/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.simulator.turn_angle = 30

        habitat_config.habitat.task.actions.update({
            "look_up": LookUpActionConfig(tilt_angle=30),
            "look_down": LookDownActionConfig(tilt_angle=30)
        })

        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig(),
        })

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov=90
        
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov=90
        
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 3.0
        
    return habitat_config
def rxr_config(path:str=RXR_CONFIG_PATH,stage:str='val_unseen', part_idx=None, img_size:int=384):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = "/habitat-data-0.2.5/scenes"
        # habitat_config.habitat.dataset.data_path = "/vepfs-cnsh4137610c2f4c/algo/user9/habitat-data-0.2.5/datasets/vlnnav/r2r/{split}/{split}.json.gz"
        if part_idx == None:
            habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/vlnnav/rxr/{split}/sampled_{split}_guide_en.json.gz"
        else:
            habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/vlnnav/rxr/{split}/sampled613_{split}_partidx.json.gz"
            habitat_config.habitat.dataset.data_path = habitat_config.habitat.dataset.data_path.replace("idx", part_idx)
        habitat_config.habitat.simulator.scene_dataset = "/habitat-data-0.2.5/scenes/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.simulator.turn_angle = 30

        habitat_config.habitat.task.actions.update({
            "look_up": LookUpActionConfig(tilt_angle=30),
            "look_down": LookDownActionConfig(tilt_angle=30)
        })

        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig()
        })

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov=90
        
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov=90
        
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 3.0
        
    return habitat_config

def rxr_w_config(path:str=RXR_CONFIG_PATH,stage:str='val_unseen', part_idx=None, img_size:int=384):
    habitat_config = habitat.get_config(path)
    with read_write(habitat_config):
        habitat_config.habitat.dataset.split = stage
        habitat_config.habitat.dataset.scenes_dir = "/habitat-data-0.2.5/scenes"
        # habitat_config.habitat.dataset.data_path = "/vepfs-cnsh4137610c2f4c/algo/user9/habitat-data-0.2.5/datasets/vlnnav/r2r/{split}/{split}.json.gz"
        if part_idx == None:
            habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/vlnnav/rxr/{split}/{split}_guide_en.json.gz"
        else:
            habitat_config.habitat.dataset.data_path = "/habitat-data-0.2.5/datasets/vlnnav/rxr/{split}/sampled613_{split}_partidx.json.gz"
            habitat_config.habitat.dataset.data_path = habitat_config.habitat.dataset.data_path.replace("idx", part_idx)
        habitat_config.habitat.simulator.scene_dataset = "/habitat-data-0.2.5/scenes/mp3d/mp3d.scene_dataset_config.json"
        habitat_config.habitat.simulator.turn_angle = 30

        habitat_config.habitat.task.actions.update({
            "look_up": LookUpActionConfig(tilt_angle=30),
            "look_down": LookDownActionConfig(tilt_angle=30)
        })

        habitat_config.habitat.task.measurements.update(
        {
            "top_down_map": TopDownMapMeasurementConfig(
                map_padding=3,
                map_resolution=1024,
                draw_source=True,
                draw_border=True,
                draw_shortest_path=False,
                draw_view_points=True,
                draw_goal_positions=True,
                draw_goal_aabbs=True,
                fog_of_war=FogOfWarConfig(
                    draw=True,
                    visibility_dist=5.0,
                    fov=79,
                ),
            ),
            "collisions": CollisionsMeasurementConfig()
        })

        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov=90
        
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.width=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.height=img_size
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.hfov=90
        
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth=5.0
        habitat_config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.normalize_depth=False
        habitat_config.habitat.task.measurements.success.success_distance = 3.0
        
    return habitat_config


def ima(refer,state,path,special_point_index):
        plt.figure()


        refer_xs, _, refer_zs = zip(*refer)
        state_xs, _, state_zs = zip(*state)

        plt.plot(refer_xs, refer_zs, marker='o', color='b', label='Refer')
        plt.plot(state_xs, state_zs, marker='o', color='r', label='State')
        
        plt.scatter(state_xs[special_point_index], state_zs[special_point_index], color='g', marker='x', s=100, label='Special Point')
        # 设置标签和图例
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.legend()

        # 保存图形为图片文件
        plt.savefig(path)

def calculate_distance(point_a, point_b):
    a = np.array(point_a).copy()
    a[1] = point_b[1]
    b = np.array(point_b)
    return np.linalg.norm(a - b)

# 插值函数
def inter(trajectory, max_distance=0.2):
    interpolated_trajectory = [trajectory[0]]
    for i in range(1, len(trajectory)):
        start_point = np.array(trajectory[i-1])
        end_point = np.array(trajectory[i])
        distance = calculate_distance(start_point, end_point)
        
        if distance > max_distance:
            num_points = int(np.ceil(distance / max_distance))
            for j in range(1, num_points):
                interpolated_point = start_point + (end_point - start_point) * (j / num_points)
                interpolated_trajectory.append(interpolated_point.tolist())
        
        interpolated_trajectory.append(end_point.tolist())
    
    return interpolated_trajectory

# 插值轨迹 a


# 找出轨迹 b 中第一个与插值后的轨迹 a 的距离超过 0.5 米的点
def find_point(trajectory_a, trajectory_b, threshold=0.5):
    for i, point_b in enumerate(trajectory_b):
        distances = [calculate_distance(point_a, point_b) for point_a in trajectory_a]
        min_distance = min(distances)
        if min_distance > threshold:
            return i
    return None
def find_min(trajectory_a, trajectory_b, index):
        
        distances = [calculate_distance(point_b, trajectory_b[index]) for point_b in trajectory_a]
        min_index = distances.index(min(distances))
        #print(distances,index)
        return min_index
def get_indices(trajectory, interpolated_trajectory):
    """
    返回原轨迹点分别在插值后轨迹中的索引列表
    """
    indices = []
    #print(trajectory,interpolated_trajectory)
    for pt in trajectory:
        for idx, ipt in enumerate(interpolated_trajectory):
            if np.allclose(ipt, pt):  # 防止数值精度误差
                indices.append(idx)
                break
    return indices
def find_next_original_traj_index(interpolated_index, original_indices):
    """
    给定插值轨迹索引和原始点在插值轨迹的下标列表，返回下一个原点在原始轨迹的下标
    """
    #print(original_indices,interpolated_index)
    for ori_traj_idx, interpolated_idx in enumerate(original_indices):
        if interpolated_idx >= interpolated_index:
            return ori_traj_idx
    return None
def find_ori(trajectory, interpolated_trajectory,interpolated_index):
    indices=get_indices(trajectory, interpolated_trajectory)
    f=find_next_original_traj_index(interpolated_index, indices)
    return f
