import shutil
import h5py
import os
import numpy as np
import torch

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "pi0_baseline/isaaclab"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    "open_door"
]

def axis_angle_from_quat(quat: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    """Convert rotations given as quaternions to axis/angle.

    Args:
        quat: The quaternion orientation in (w, x, y, z). Shape is (..., 4).
        eps: The tolerance for Taylor approximation. Defaults to 1.0e-6.

    Returns:
        Rotations given as a vector in axis angle form. Shape is (..., 3).
        The vector's magnitude is the angle turned anti-clockwise in radians around the vector's direction.

    Reference:
        https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py#L526-L554
    """
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]
    # Thus, axis-angle is [q_x, q_y, q_z] / (sin(theta/2) / theta)
    # When theta = 0, (sin(theta/2) / theta) is undefined
    # However, as theta --> 0, we can use the Taylor approximation 1/2 - theta^2 / 48
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    # check whether to apply Taylor approximation
    sin_half_angles_over_angles = torch.where(
        angle.abs() > eps, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)

def main(data_dir: str, *, push_to_hub: bool = False):
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=30,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_data_list = os.listdir(f"{data_dir}/{raw_dataset_name}")
        for raw_data in raw_data_list:
            raw_data = h5py.File(f"{data_dir}/{raw_dataset_name}/{raw_data}", "r")
            demo_datas = raw_data["/data"]
            # using tqdm to show the progress
            for demo in tqdm(demo_datas):
                processed_actions = demo_datas[demo]["processed_actions"]
                actions = demo_datas[demo]["actions"]
                pos_states = demo_datas[demo]['obs']['ee_pos']
                quat_states = demo_datas[demo]['obs']['ee_quat']
                table_cam = demo_datas[demo]["obs"]["table_cam"]
                wrist_cam = demo_datas[demo]["obs"]["wrist_cam"]
                for i in tqdm(range(len(actions))):
                    if i == 0:
                        continue
                    image = table_cam[i]
                    wrist_image = wrist_cam[i]
                    
                    # reshape the image to (256, 256, 3)
                    image.resize((256, 256, 3))
                    # wrist_image = wrist_image.resize((256, 256, 3))
                    # process the action
                    action = actions[i]
                    # change quat to axis angle
                    position = action[0:3]
                    quat = action[3:7]
                    quat = torch.from_numpy(quat).unsqueeze(0)
                    axis_angle = axis_angle_from_quat(quat)
                    axis_angle = axis_angle.squeeze(0).numpy()
                    if action[7] == 1: # this means the gripper is open. but we need to change this to 0.0 for training model.
                        gripper_action = 0.0
                    else:
                        gripper_action = 1.0
                    new_action = np.concatenate([position, axis_angle, [gripper_action]], axis=0)
                    # change new action to float32
                    new_action = new_action.astype(np.float32)
                    # process the state
                    obs_pos = pos_states[i]
                    obs_quat = quat_states[i]
                    obs_quat = torch.from_numpy(obs_quat).unsqueeze(0)
                    obs_axis_angle = axis_angle_from_quat(obs_quat)
                    obs_axis_angle = obs_axis_angle.squeeze(0).numpy()
                    if action[7] == 1:
                        obs_gripper_state = [0.04, 0.04]
                    else:
                        obs_gripper_state = [0.0, 0.0]
                    new_state = np.concatenate([obs_pos, obs_axis_angle, obs_gripper_state], axis=0)
                    new_state = new_state.astype(np.float32)
                    
                    task_inst = "open the right door of white drawer "
                    dataset.add_frame(
                        {
                            "image": image,
                            "wrist_image": wrist_image,
                            "state": new_state,
                            "actions": new_action,
                            "task": task_inst,
                        }
                    )
                dataset.save_episode()

    print('dataset saved successfully')                



if __name__ == "__main__":
    tyro.cli(main)