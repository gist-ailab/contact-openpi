"""
h5py 형식으로 저장된 Libero 데이터를 읽고 사용하는 예시 스크립트.

이 스크립트는 convert_libero_to_h5py.py로 생성된 h5py 파일을 읽어서
다양한 방식으로 데이터를 활용하는 방법을 보여줍니다.

Usage:
uv run examples/libero/read_h5py_dataset.py --file_path libero_dataset.h5

특정 에피소드/스텝 조회:
uv run examples/libero/read_h5py_dataset.py --file_path libero_dataset.h5 --episode_idx 0 --step_idx 5

데이터 통계 보기:
uv run examples/libero/read_h5py_dataset.py --file_path libero_dataset.h5 --show_stats

필요한 패키지:
`uv pip install h5py matplotlib numpy`
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import tyro
from typing import Optional, List, Dict, Any
import os

def print_dataset_info(file_path: str):
    """데이터셋의 기본 정보를 출력"""
    with h5py.File(file_path, 'r') as f:
        print("=== Dataset Information ===")
        print(f"File: {file_path}")
        print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
        
        print("\n--- Metadata ---")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        print("\n--- Structure ---")
        episodes_group = f['episodes']
        num_episodes = len(episodes_group.keys())
        print(f"  Number of episodes: {num_episodes}")
        
        # 첫 번째 에피소드 정보
        if num_episodes > 0:
            first_episode = list(episodes_group.keys())[0]
            first_episode_group = episodes_group[first_episode]
            num_steps = first_episode_group.attrs['num_steps']
            print(f"  Steps per episode (first episode): {num_steps}")
            
            # 첫 번째 스텝 정보
            if num_steps > 0:
                first_step = list(first_episode_group.keys())[0]
                first_step_group = first_episode_group[first_step]
                print(f"  Data keys in each step: {list(first_step_group.keys())}")

def read_episode_step(file_path: str, episode_idx: int, step_idx: int) -> Dict[str, Any]:
    """특정 에피소드와 스텝의 데이터를 읽어서 반환"""
    with h5py.File(file_path, 'r') as f:
        episode_group = f[f'episodes/episode_{episode_idx:06d}']
        step_group = episode_group[f'step_{step_idx:06d}']
        
        data = {
            'base_image': step_group['observation']['image'][:],
            'wrist_image': step_group['observation']['wrist_image'][:],
            'state': step_group['observation']['state'][:],
            'action': step_group['action'][:],
            'task': step_group['language_instruction'][()],
            'episode_metadata': dict(episode_group.attrs),
            'step_metadata': dict(step_group.attrs)
        }
        
        return data

def visualize_step(data: Dict[str, Any], episode_idx: int, step_idx: int, save_path: Optional[str] = None):
    """스텝 데이터를 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Base image
    axes[0].imshow(data['base_image'])
    axes[0].set_title(f'Base Camera View\nEpisode {episode_idx}, Step {step_idx}')
    axes[0].axis('off')
    
    # Wrist image
    axes[1].imshow(data['wrist_image'])
    axes[1].set_title(f'Wrist Camera View\nEpisode {episode_idx}, Step {step_idx}')
    axes[1].axis('off')
    
    # Task description
    task_text = data['task'][:100] + "..." if len(data['task']) > 100 else data['task']
    fig.suptitle(f'Task: {task_text}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()
    plt.close()

def get_dataset_statistics(file_path: str) -> Dict[str, Any]:
    """데이터셋의 통계 정보를 계산"""
    stats = {
        'total_episodes': 0,
        'total_steps': 0,
        'episode_lengths': [],
        'state_ranges': {'min': [], 'max': []},
        'action_ranges': {'min': [], 'max': []},
        'task_lengths': []
    }
    
    with h5py.File(file_path, 'r') as f:
        episodes_group = f['episodes']
        
        for episode_name in episodes_group.keys():
            episode_group = episodes_group[episode_name]
            num_steps = episode_group.attrs['num_steps']
            stats['total_episodes'] += 1
            stats['total_steps'] += num_steps
            stats['episode_lengths'].append(num_steps)
            
            # 각 스텝의 통계 수집
            for step_name in episode_group.keys():
                step_group = episode_group[step_name]
                
                # State 통계
                state = step_group['state'][:]
                stats['state_ranges']['min'].append(state.min())
                stats['state_ranges']['max'].append(state.max())
                
                # Action 통계
                action = step_group['action'][:]
                stats['action_ranges']['min'].append(action.min())
                stats['action_ranges']['max'].append(action.max())
                
                # Task 길이
                task = step_group['task'][()]
                stats['task_lengths'].append(len(task))
    
    # 통계 계산
    stats['episode_lengths'] = np.array(stats['episode_lengths'])
    stats['state_ranges']['min'] = np.array(stats['state_ranges']['min'])
    stats['state_ranges']['max'] = np.array(stats['state_ranges']['max'])
    stats['action_ranges']['min'] = np.array(stats['action_ranges']['min'])
    stats['action_ranges']['max'] = np.array(stats['action_ranges']['max'])
    stats['task_lengths'] = np.array(stats['task_lengths'])
    
    return stats

def print_statistics(stats: Dict[str, Any]):
    """통계 정보를 출력"""
    print("=== Dataset Statistics ===")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Total steps: {stats['total_steps']}")
    
    print(f"\nEpisode lengths:")
    print(f"  Mean: {stats['episode_lengths'].mean():.2f}")
    print(f"  Std: {stats['episode_lengths'].std():.2f}")
    print(f"  Min: {stats['episode_lengths'].min()}")
    print(f"  Max: {stats['episode_lengths'].max()}")
    
    print(f"\nState ranges:")
    print(f"  Min: {stats['state_ranges']['min'].min():.4f}")
    print(f"  Max: {stats['state_ranges']['max'].max():.4f}")
    
    print(f"\nAction ranges:")
    print(f"  Min: {stats['action_ranges']['min'].min():.4f}")
    print(f"  Max: {stats['action_ranges']['max'].max():.4f}")
    
    print(f"\nTask description lengths:")
    print(f"  Mean: {stats['task_lengths'].mean():.2f}")
    print(f"  Std: {stats['task_lengths'].std():.2f}")
    print(f"  Min: {stats['task_lengths'].min()}")
    print(f"  Max: {stats['task_lengths'].max()}")

def create_data_loader(file_path: str, batch_size: int = 32, shuffle: bool = True):
    """데이터 로더 생성 (PyTorch 스타일)"""
    with h5py.File(file_path, 'r') as f:
        episodes_group = f['episodes']
        episode_names = list(episodes_group.keys())
        
        if shuffle:
            np.random.shuffle(episode_names)
        
        # 모든 스텝을 (episode_idx, step_idx) 형태로 수집
        all_steps = []
        for episode_name in episode_names:
            episode_group = episodes_group[episode_name]
            episode_idx = episode_group.attrs['episode_id']
            
            for step_name in episode_group.keys():
                step_idx = int(step_name.split('_')[1])
                all_steps.append((episode_idx, step_idx))
        
        if shuffle:
            np.random.shuffle(all_steps)
        
        # 배치 생성
        for i in range(0, len(all_steps), batch_size):
            batch_steps = all_steps[i:i + batch_size]
            batch_data = []
            
            for episode_idx, step_idx in batch_steps:
                data = read_episode_step(file_path, episode_idx, step_idx)
                batch_data.append(data)
            
            yield batch_data

def main(
    file_path: str,
    episode_idx: Optional[int] = None,
    step_idx: Optional[int] = None,
    show_stats: bool = False,
    visualize: bool = False,
    batch_size: int = 1,
    num_batches: int = 1
):
    """
    h5py 데이터셋을 읽고 분석
    
    Args:
        file_path: h5py 파일 경로
        episode_idx: 조회할 에피소드 인덱스
        step_idx: 조회할 스텝 인덱스
        show_stats: 통계 정보 출력 여부
        visualize: 이미지 시각화 여부
        batch_size: 배치 크기
        num_batches: 처리할 배치 수
    """
    print(f"Reading h5py dataset: {file_path}")
    
    # 기본 정보 출력
    print_dataset_info(file_path)
    
    # 통계 정보 출력 (요청된 경우)
    if show_stats:
        print("\n" + "="*50)
        stats = get_dataset_statistics(file_path)
        print_statistics(stats)
    
    # 특정 에피소드/스텝 조회
    if episode_idx is not None and step_idx is not None:
        print(f"\n" + "="*50)
        print(f"Reading Episode {episode_idx}, Step {step_idx}")
        
        try:
            data = read_episode_step(file_path, episode_idx, step_idx)
            print(f"Task: {data['task']}")
            print(f"State shape: {data['state'].shape}")
            print(f"Action shape: {data['action'].shape}")
            print(f"Base image shape: {data['base_image'].shape}")
            print(f"Wrist image shape: {data['wrist_image'].shape}")
            
            if visualize:
                visualize_step(data, episode_idx, step_idx)
                
        except KeyError as e:
            print(f"Error: {e}")
            print("Episode or step not found in dataset")
    
    # 배치 데이터 로더 테스트
    print(f"\n" + "="*50)
    print(f"Testing batch data loader (batch_size={batch_size}, num_batches={num_batches})")
    
    batch_count = 0
    for batch in create_data_loader(file_path, batch_size=batch_size, shuffle=True):
        print(f"\nBatch {batch_count + 1}:")
        print(f"  Batch size: {len(batch)}")
        print(f"  First item task: {batch[0]['task'][:50]}...")
        print(f"  State shapes: {[item['state'].shape for item in batch]}")
        print(f"  Action shapes: {[item['action'].shape for item in batch]}")
        
        batch_count += 1
        if batch_count >= num_batches:
            break

if __name__ == "__main__":
    tyro.cli(main) 