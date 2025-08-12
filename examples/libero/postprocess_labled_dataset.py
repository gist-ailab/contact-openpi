import shutil
import h5py
import os
import numpy as np
import cv2
import tyro

REPO_NAME = "obj_centric_vla/libero"  # Name of the output dataset, also used for the Hugging Face Hub
RAW_DATASET_NAMES = [
    # "libero_10_no_noops",
    
    "libero_goal_no_noops",
    
    # "libero_object_no_noops",
    # "libero_spatial_no_noops",
    
    # "processed_with_bbox",
]  # For simplicity we will combine multiple Libero datasets into one training dataset


def copy_h5py_group(src_group, dst_group):
    """Recursively copy an h5py group and its contents"""
    for key in src_group.keys():
        if isinstance(src_group[key], h5py.Group):
            # Create subgroup in destination
            dst_group.create_group(key)
            copy_h5py_group(src_group[key], dst_group[key])
        else:
            # Copy dataset
            src_group.copy(key, dst_group)


def process_and_save_dataset(data_dir: str, output_dir: str, *, visualize: bool = False):
    """Process datasets and save with bbox annotations"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for raw_dataset_name in RAW_DATASET_NAMES:
        raw_data_list = os.listdir(f"{data_dir}/h5_dataset/{raw_dataset_name}")
        
        for raw_data in raw_data_list:
            input_path = f"{data_dir}/h5_dataset/{raw_dataset_name}/{raw_data}"
            output_path = f"{output_dir}/{raw_dataset_name}_{raw_data}"
            
            print(f"Processing {input_path} -> {output_path}")
            
            # Open source dataset
            with h5py.File(input_path, "r") as raw_dataset:
                # Create new dataset
                with h5py.File(output_path, "w") as new_dataset:
                    # Copy the entire structure
                    copy_h5py_group(raw_dataset, new_dataset)
                    
                    # Now add bbox annotations to each observation
                    episodes_data = new_dataset["episodes"]
                    for episode in episodes_data:
                        episode_data = episodes_data[episode]
                        for step in episode_data:
                            step_data = episode_data[step]
                            observation = step_data['observation']
                            # Get point labels
                            point_label = observation['point_label'][()]  # (P, 3), z is 0, P is the number of points
                            point_label = point_label.astype(int)
                            
                            # Convert point labels to bboxes
                            # point label is (x, y, z) and bbox is (y_min, x_min, y_max, x_max)
                            # make point label to bbox. x and y should be the center of the bbox. width and height should be 10px
                            bboxs = np.zeros((point_label.shape[0], 4))
                            bboxs[:, 0] = point_label[:, 1] - 5  # y_min
                            bboxs[:, 1] = point_label[:, 0] - 5  # x_min
                            bboxs[:, 2] = point_label[:, 1] + 5  # y_max
                            bboxs[:, 3] = point_label[:, 0] + 5  # x_max
                            
                            # Ensure coordinates are within image bounds
                            bboxs[:, 0] = np.clip(bboxs[:, 0], 0, 255)  # y_min
                            bboxs[:, 1] = np.clip(bboxs[:, 1], 0, 255)  # x_min
                            bboxs[:, 2] = np.clip(bboxs[:, 2], 0, 255)  # y_max
                            bboxs[:, 3] = np.clip(bboxs[:, 3], 0, 255)  # x_max
                            
                            # make bboxs to paligemma format
                            # first normalize bbox to 0-1023
                            paligemma_format = ""
                            if len(bboxs) == 1:
                                y_min_norm = int((bboxs[0, 0] / 256) * 1023)
                                x_min_norm = int((bboxs[0, 1] / 256) * 1023)
                                y_max_norm = int((bboxs[0, 2] / 256) * 1023)
                                x_max_norm = int((bboxs[0, 3] / 256) * 1023)
                                paligemma_format += f"<loc{y_min_norm:04d}><loc{x_min_norm:04d}><loc{y_max_norm:04d}><loc{x_max_norm:04d}>"
                            else:
                                for i, bbox in enumerate(bboxs):
                                    y_min_norm = int((bbox[0] / 256) * 1023)
                                    x_min_norm = int((bbox[1] / 256) * 1023)
                                    y_max_norm = int((bbox[2] / 256) * 1023)
                                    x_max_norm = int((bbox[3] / 256) * 1023)
                                    if i == len(bboxs) - 1:
                                        paligemma_format += f"<loc{y_min_norm:04d}><loc{x_min_norm:04d}><loc{y_max_norm:04d}><loc{x_max_norm:04d}>"
                                    else:
                                        paligemma_format += f"<loc{y_min_norm:04d}><loc{x_min_norm:04d}><loc{y_max_norm:04d}><loc{x_max_norm:04d}>;"
                            
                            # Add bbox data to observation
                            observation.create_dataset('bboxs', data=paligemma_format.encode())
                            
                            if visualize:
                                image = observation['image'][()]
                                image = image.astype(np.uint8)
                                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                
                                # Resize image to make it larger
                                scale_factor = 2.0  # 2배 크기로 확대
                                new_width = int(image.shape[1] * scale_factor)
                                new_height = int(image.shape[0] * scale_factor)
                                image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                                
                                # Scale bbox coordinates accordingly
                                for bbox in bboxs:
                                    # cv2.rectangle expects (x, y) coordinates
                                    # bbox format: (y_min, x_min, y_max, x_max)
                                    pt1 = (int(bbox[1] * scale_factor), int(bbox[0] * scale_factor))  # (x_min, y_min)
                                    pt2 = (int(bbox[3] * scale_factor), int(bbox[2] * scale_factor))  # (x_max, y_max)
                                    cv2.rectangle(image_resized, pt1, pt2, (0, 0, 255), 3)  # 선 두께도 증가
                                
                                # Create a named window and set its size
                                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                                cv2.resizeWindow('image', new_width, new_height)
                                cv2.imshow('image', image_resized)
                                cv2.waitKey(0)
                                cv2.destroyAllWindows()
            
            print(f"Saved processed dataset to {output_path}")


def main(data_dir: str, output_dir: str = None, *, visualize: bool = False):
    """Main function to process and save datasets with bbox annotations"""
    
    if output_dir is None:
        output_dir = f"{data_dir}/h5_dataset/libero_goal_no_noops_w_bbox"
    
    process_and_save_dataset(data_dir, output_dir, visualize=visualize)
    print(f"All datasets processed and saved to {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)