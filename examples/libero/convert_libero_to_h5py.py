"""
RLDS 형식의 Libero 데이터를 h5py 형식으로 변환하는 스크립트.

기존 RLDS 형식:
episodes/
├── episode_000000/
│   ├── step_000000/
│   │   ├── action
│   │   ├── is_terminal
│   │   ├── is_last
│   │   ├── is_first
│   │   ├── language_instruction
│   │   ├── reward
│   │   ├── discount
│   │   └── observation/
│   │       ├── image
│   │       ├── wrist_image
│   │       ├── state
│   │       └── joint_state
│   └── step_000001/
│       └── ...
└── episode_000001/
    └── ...

h5py 형식:
episodes/
├── episode_000000/
│   ├── step_000000/
│   │   ├── action
│   │   ├── is_terminal
│   │   ├── is_last
│   │   ├── is_first
│   │   ├── language_instruction
│   │   ├── reward
│   │   ├── discount
│   │   └── observation/
│   │       ├── image
│   │       ├── wrist_image
│   │       ├── state
│   │       └── joint_state
│   └── step_000001/
│       └── ...
└── episode_000001/
    └── ...


Usage:
uv run examples/libero/convert_libero_to_h5py.py --data_dir /path/to/your/data --output_file libero_dataset.h5

필요한 패키지:
`uv pip install tensorflow tensorflow_datasets h5py`
"""

import os
import numpy as np
import h5py
import time
import tensorflow_datasets as tfds
import tyro
from typing import Optional

RAW_DATASET_NAMES = [
    "libero_goal_no_noops",
    # "libero_10_no_noops",
    # "libero_object_no_noops",
    # "libero_spatial_no_noops",
]


def main(
    data_dir: str, 
    output_file: str = "libero_dataset.h5",
    *, 
    max_episodes: Optional[int] = None, 
    max_steps: Optional[int] = None,
    compression: bool = False
):
    """
    RLDS 형식의 Libero 데이터를 h5py 형식으로 변환
    
    Args:
        data_dir: RLDS 데이터가 저장된 디렉토리 경로
        output_file: 출력 h5py 파일 경로
        max_episodes: 처리할 최대 에피소드 수 (None이면 모든 에피소드)
        max_steps: 에피소드당 최대 스텝 수 (None이면 모든 스텝)
        compression: 이미지 압축 사용 여부
    """
    print(f"Converting RLDS data from {data_dir} to h5py format: {output_file}")
    
    # h5py 파일 생성
    with h5py.File(output_file, 'w') as f:
        # 메타데이터 저장
        f.attrs['dataset_name'] = 'Libero Dataset (RLDS Structure)'
        f.attrs['robot_type'] = 'panda'
        f.attrs['fps'] = 10
        f.attrs['image_shape'] = (256, 256, 3)
        f.attrs['state_dim'] = 8
        f.attrs['joint_state_dim'] = 7
        f.attrs['action_dim'] = 7
        f.attrs['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
        f.attrs['compression'] = compression
        f.attrs['structure'] = 'Original RLDS structure preserved'
        
        # 에피소드 그룹 생성
        episodes_group = f.create_group('episodes')
        episodes_group.attrs['description'] = 'Robot episodes with observations and actions'
        
        episode_count = 0
        total_steps = 0
        
        # 각 원본 데이터셋 처리
        for raw_dataset_name in RAW_DATASET_NAMES:
            print(f"\nProcessing dataset: {raw_dataset_name}")
            
            raw_dataset = tfds.load(raw_dataset_name, data_dir=data_dir, split="train")
            print(f"Dataset cardinality: {raw_dataset.cardinality()}")
            
            for episode in raw_dataset:
                if max_episodes is not None and episode_count >= max_episodes:
                    break
                
                
                print(f"\nProcessing episode {episode_count}")
                
                # 에피소드 그룹 생성
                episode_group = episodes_group.create_group(f'episode_{episode_count:06d}')
                episode_group.attrs['dataset_source'] = raw_dataset_name
                episode_group.attrs['episode_id'] = episode_count
                
                step_count = 0
                episode_data = []
                
                # 에피소드 메타데이터 저장
                if "episode_metadata" in episode:
                    episode_metadata = episode["episode_metadata"]
                    if "file_path" in episode_metadata:
                        episode_group.attrs['file_path'] = episode_metadata["file_path"].numpy().decode()
                
                for step in episode["steps"].as_numpy_iterator():
                    if max_steps is not None and step_count >= max_steps:
                        break
                    
                    # 스텝 그룹 생성
                    step_group = episode_group.create_group(f'step_{step_count:06d}')
                    
                    # 1. Action 데이터 저장
                    step_group.create_dataset('action', data=step["action"])
                    
                    # 2. Terminal/Last/First 플래그 저장
                    step_group.create_dataset('is_terminal', data=step["is_terminal"])
                    step_group.create_dataset('is_last', data=step["is_last"])
                    step_group.create_dataset('is_first', data=step["is_first"])
                    
                    # 3. Language instruction 저장
                    step_group.create_dataset('language_instruction', 
                                            data=step["language_instruction"].decode(), 
                                            dtype=h5py.special_dtype(vlen=str))
                    
                    # 4. Observation 데이터 저장
                    observation_group = step_group.create_group('observation')
                    
                    # 4.1 이미지 데이터 저장
                    if compression:
                        observation_group.create_dataset('image', data=step["observation"]["image"], 
                                                        compression='gzip', compression_opts=6)
                        observation_group.create_dataset('wrist_image', data=step["observation"]["wrist_image"], 
                                                        compression='gzip', compression_opts=6)
                    else:
                        observation_group.create_dataset('image', data=step["observation"]["image"])
                        observation_group.create_dataset('wrist_image', data=step["observation"]["wrist_image"])
                    
                    # 4.2 상태 데이터 저장
                    observation_group.create_dataset('state', data=step["observation"]["state"])
                    observation_group.create_dataset('joint_state', data=step["observation"]["joint_state"])
                    
                    # 5. Reward/Discount 데이터 저장
                    step_group.create_dataset('reward', data=step["reward"])
                    step_group.create_dataset('discount', data=step["discount"])
                    
                    # 진행 상황 출력
                    task = step["language_instruction"].decode()
                    # print(f"  Step {step_count}: {task[:50]}...")
                    
                    step_count += 1
                    total_steps += 1
                
                episode_group.attrs['num_steps'] = step_count
                episode_count += 1
                
                print(f"  Saved {step_count} steps for episode {episode_count-1}")
        
        # 전체 통계 저장
        f.attrs['total_episodes'] = episode_count
        f.attrs['total_steps'] = total_steps
    
    print(f"\nConversion completed!")
    print(f"Output file: {output_file}")
    print(f"Total episodes: {episode_count}")
    print(f"Total steps: {total_steps}")
    
    # 파일 크기 정보 출력
    file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    print(f"File size: {file_size:.2f} MB")

if __name__ == "__main__":
    tyro.cli(main) 