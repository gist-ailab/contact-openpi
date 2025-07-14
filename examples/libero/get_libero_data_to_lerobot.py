"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/get_libero_data_to_lerobot.py --data_dir /path/to/your/data

To visualize images during processing:
uv run examples/libero/get_libero_data_to_lerobot.py --data_dir /path/to/your/data --visualize --max_episodes 2 --max_steps 5

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/get_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')
import os

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro

REPO_NAME = "obj_centric_vla/libero"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    # "libero_10_no_noops",
    "libero_goal_no_noops",
    # "libero_object_no_noops",
    # "libero_spatial_no_noops",
]  # For simplicity we will combine multiple Libero datasets into one training dataset

def visualize_images(base_image, wrist_image, episode_idx, step_idx, task_description):
    """Visualize images using matplotlib with better layout and information display"""
    # Ensure images are in the correct format (0-1 float or 0-255 uint8)
    if base_image.dtype != np.uint8:
        if base_image.max() <= 1.0:
            base_image = (base_image * 255).astype(np.uint8)
        else:
            base_image = base_image.astype(np.uint8)
    if wrist_image.dtype != np.uint8:
        if wrist_image.max() <= 1.0:
            wrist_image = (wrist_image * 255).astype(np.uint8)
        else:
            wrist_image = wrist_image.astype(np.uint8)
    
    try:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display base image
        ax1.imshow(base_image)
        ax1.set_title(f'Base Camera View\nEpisode {episode_idx}, Step {step_idx}', fontsize=12)
        ax1.axis('off')
        
        # Display wrist image
        ax2.imshow(wrist_image)
        ax2.set_title(f'Wrist Camera View\nEpisode {episode_idx}, Step {step_idx}', fontsize=12)
        ax2.axis('off')
        
        # Add task description as suptitle
        fig.suptitle(f'Task: {task_description[:100]}{"..." if len(task_description) > 100 else ""}', 
                    fontsize=14, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save image
        output_dir = "visualization_output"
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/ep{episode_idx}_step{step_idx}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Image saved to: {filename}")
        
        # Show the plot
        plt.show()
        # print("Press Enter to continue to next image...")
        # input()
        
        # Close the figure to free memory
        plt.close(fig)
        return True
        
    except Exception as e:
        print(f"Matplotlib visualization failed: {e}")
        print("Skipping visualization...")
        return False


def main(data_dir: str, *, push_to_hub: bool = False, visualize: bool = False, max_episodes: int = 3, max_steps: int = 10):
    # Initialize visualization
    if visualize:
        print("Visualization mode enabled. Images will be displayed using matplotlib.")
        print("Images will also be saved to 'visualization_output' directory.")
        print("Press Enter after each image to continue to the next one.")
    
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)
    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
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

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
    episode_count = 0
    for raw_dataset_name in RAW_DATASET_NAMES:

        raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
        print(raw_dataset)
        print(raw_dataset.cardinality())
        print('--------------------------------')
        count=0
        for episode in raw_dataset:
            print('--------------------------------')
            if episode_count >= max_episodes:
                break
                
            step_count = 0
            for step in episode["steps"].as_numpy_iterator():
                if step_count >= max_steps:
                    break
                    
                # Extract data
                base_image = step["observation"]["image"]
                wrist_image = step["observation"]["wrist_image"]
                state = step["observation"]["state"]
                action = step["action"]
                task = step["language_instruction"].decode()
                print('task: ',task)
                # Visualize images if requested
                if visualize:
                    print(f"\n=== Episode {episode_count}, Step {step_count} ===")
                    print(f"Task: {task}")
                    print(f"State shape: {state.shape}, Action shape: {action.shape}")
                    visualize_images(base_image, wrist_image, episode_count, step_count, task)
                
            #     dataset.add_frame(
            #         {
            #             "image": base_image,
            #             "wrist_image": wrist_image,
            #             "state": state,
            #             "actions": action,
            #             "task": task,
            #         }
            #     )
                step_count += 1
            count+=1
                
            # dataset.save_episode()
            episode_count += 1
        print('count: ',count)
        if episode_count >= max_episodes:
            break

    # Consolidate the dataset, skip computing stats since we will do that later
    # dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
