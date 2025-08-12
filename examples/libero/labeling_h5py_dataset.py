"""
h5py 파일을 불러와서 매 스텝마다 이미지를 시각화하는 스크립트.

이 스크립트는 convert_libero_to_h5py.py로 생성된 h5py 파일을 읽어서
각 스텝의 이미지들을 시각화하고 라벨링을 위한 인터페이스를 제공합니다.

Usage:
uv run examples/libero/visualize_h5py_dataset.py --file_path modified_libero_rlds_ctp/h5_dataset/libero_dataset.h5

특정 에피소드만 시각화:
uv run examples/libero/visualize_h5py_dataset.py --file_path modified_libero_rlds_ctp/h5_dataset/libero_dataset.h5 --episode_idx 0

특정 스텝 범위만 시각화:
uv run examples/libero/visualize_h5py_dataset.py --file_path modified_libero_rlds_ctp/h5_dataset/libero_dataset.h5 --episode_idx 0 --start_step 0 --end_step 10

필요한 패키지:
`uv pip install h5py matplotlib numpy opencv-python`
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, TextBox
import tyro
from typing import Optional, Dict, Any, List
import os
import cv2
import threading
import time
from scipy.linalg import lstsq

class H5pyVisualizer:
    def __init__(self, file_path: str, save_path: str):
        self.file_path = file_path
        self.save_path = save_path
        # 읽기/쓰기 모드로 파일 열기
        self.f = h5py.File(file_path, 'r+')
        self.episodes_group = self.f['episodes']
        self.episode_names = sorted(list(self.episodes_group.keys()))
        self.current_episode_idx = 0
        self.current_step_idx = 0
        self.labels = {}  # 라벨링 결과 저장
        self.fig = None
        self.current_step = 0
        self.running = True
        self.episode_idx = 0
        self.start_step = 0
        self.end_step = 0
        self.waiting_for_input = False
        self.labeling_mode = False  # 라벨링 모드 상태
        self.clicked_points = []  # 클릭한 점들 저장
        self.axes = None  # 현재 axes 참조
        self.dlt_mode = False  # DLT 모드 상태
        self.dlt_points_2d = []  # DLT용 2D 점들
        self.dlt_points_3d = []  # DLT용 3D 점들
        self.dlt_transform_matrix = None  # DLT 변환 행렬
        self.auto_labeling_enabled = False  # 자동 라벨링 활성화 상태
        self.copy_to_all_steps_mode = False  # 모든 스텝에 복사 모드 상태
        self.copy_to_past_steps_mode = False  # 이전 스텝에 복사 모드 상태
        self.auto_labeling_edit_mode = False  # 자동 라벨링 편집 모드 상태
        self.all_dlt_points_2d = []  # 모든 DLT 2D 점들 (에피소드 전체)
        self.all_dlt_points_3d = []  # 모든 DLT 3D 점들 (에피소드 전체)
        self.all_dlt_steps = []  # 각 DLT 점이 수집된 스텝들
        
    def get_episode_info(self, episode_idx: int) -> Dict[str, Any]:
        """에피소드 정보를 가져옴"""
        episode_name = self.episode_names[episode_idx]
        episode_group = self.episodes_group[episode_name]
        
        info = {
            'episode_id': episode_group.attrs.get('episode_id', episode_idx),
            'dataset_source': episode_group.attrs.get('dataset_source', 'unknown'),
            'num_steps': episode_group.attrs.get('num_steps', 0),
            'file_path': episode_group.attrs.get('file_path', 'unknown')
        }
        return info
    
    def get_step_data(self, episode_idx: int, step_idx: int) -> Dict[str, Any]:
        """특정 스텝의 데이터를 가져옴"""
        episode_name = self.episode_names[episode_idx]
        episode_group = self.episodes_group[episode_name]
        step_name = f'step_{step_idx:06d}'
        
        if step_name not in episode_group:
            raise ValueError(f"Step {step_idx} not found in episode {episode_idx}")
        
        step_group = episode_group[step_name]
        
        data = {
            'action': step_group['action'][:],
            'is_terminal': step_group['is_terminal'][()],
            'is_last': step_group['is_last'][()],
            'is_first': step_group['is_first'][()],
            'language_instruction': step_group['language_instruction'][()],
            'reward': step_group['reward'][()],
            'discount': step_group['discount'][()],
            'observation': {
                'image': step_group['observation/image'][:],
                'wrist_image': step_group['observation/wrist_image'][:],
                'state': step_group['observation/state'][:],
                'joint_state': step_group['observation/joint_state'][:]
            }
        }
        
        return data
    
    def visualize_step(self, episode_idx: int, step_idx: int, save_path: Optional[str] = None):
        """특정 스텝을 시각화"""
        try:
            data = self.get_step_data(episode_idx, step_idx)
        except ValueError as e:
            print(f"Error: {e}")
            return
        
        # 이미지 정규화 (0-1 범위로)
        base_image = data['observation']['image'].astype(np.float32) / 255.0
        wrist_image = data['observation']['wrist_image'].astype(np.float32) / 255.0
        
        # 기존 figure가 있으면 닫기
        if self.fig is not None:
            plt.close(self.fig)
        
        # 시각화
        self.fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        self.fig.suptitle(f'Episode {episode_idx}, Step {step_idx}', fontsize=20, fontweight='bold')
        
        # Base image
        axes[0, 0].imshow(base_image)
        axes[0, 0].set_title('Base Camera View', fontsize=18)
        axes[0, 0].axis('off')
        
        # Wrist image
        axes[0, 1].imshow(wrist_image)
        axes[0, 1].set_title('Wrist Camera View', fontsize=18)
        axes[0, 1].axis('off')
        
        # axes 참조 저장
        self.axes = axes
        
        # 점들 표시
        if self.labeling_mode:
            # 라벨링 모드: 현재 편집 중인 점들만 표시
            points_to_show = self.clicked_points
        else:
            # 일반 모드: h5py에서 로드한 기존 점들만 표시
            points_to_show = self.load_point_labels_from_h5(episode_idx, step_idx)
        
        # point_image 마스크 오버레이 표시 (라벨링 모드가 아닐 때만)
        if not self.labeling_mode:
            base_point_image, wrist_point_image = self.load_point_images_from_h5(episode_idx, step_idx)
            
            if base_point_image is not None:
                # base 이미지에 마스크 오버레이 - 더 선명하게 표시
                # 마스크가 0이 아닌 부분만 표시하도록 마스킹
                base_mask = base_point_image > 0
                if np.any(base_mask):
                    # 빨간색으로 점들만 표시
                    overlay = np.zeros((*base_point_image.shape, 4), dtype=np.float32)
                    overlay[base_mask, 0] = 1.0  # 빨간색
                    overlay[base_mask, 3] = 0.7  # 알파값
                    axes[0, 0].imshow(overlay, alpha=0.7)
            
            if wrist_point_image is not None:
                # wrist 이미지에 마스크 오버레이 - 더 선명하게 표시
                # 마스크가 0이 아닌 부분만 표시하도록 마스킹
                wrist_mask = wrist_point_image > 0
                if np.any(wrist_mask):
                    # 파란색으로 점들만 표시
                    overlay = np.zeros((*wrist_point_image.shape, 4), dtype=np.float32)
                    overlay[wrist_mask, 2] = 1.0  # 파란색
                    overlay[wrist_mask, 3] = 0.7  # 알파값
                    axes[0, 1].imshow(overlay, alpha=0.7)
        
        for i, point in enumerate(points_to_show):
            # 자동 라벨링된 점인지 확인
            is_auto_labeled = point.get('auto_labeled', False)
            
            if point['image_type'] == 'base':
                # 자동 라벨링된 점은 다른 색상과 모양으로 표시
                if is_auto_labeled:
                    axes[0, 0].plot(point['x'], point['y'], '^', color='orange', markersize=12, 
                                   markeredgecolor='white', markeredgewidth=2)
                    axes[0, 0].text(point['x'] + 5, point['y'] + 5, f"A{i+1}", 
                                   color='white', fontsize=12, weight='bold',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor='orange', alpha=0.8))
                else:
                    axes[0, 0].plot(point['x'], point['y'], 'o', color='red', markersize=10, 
                                   markeredgecolor='white', markeredgewidth=2)
                    axes[0, 0].text(point['x'] + 5, point['y'] + 5, str(i+1), 
                                   color='white', fontsize=12, weight='bold',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.8))
            elif point['image_type'] == 'wrist':
                # 자동 라벨링된 점은 다른 색상과 모양으로 표시
                if is_auto_labeled:
                    axes[0, 1].plot(point['x'], point['y'], '^', color='yellow', markersize=12, 
                                   markeredgecolor='white', markeredgewidth=2)
                    axes[0, 1].text(point['x'] + 5, point['y'] + 5, f"A{i+1}", 
                                   color='white', fontsize=12, weight='bold',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.8))
                else:
                    axes[0, 1].plot(point['x'], point['y'], 'o', color='blue', markersize=10, 
                                   markeredgecolor='white', markeredgewidth=2)
                    axes[0, 1].text(point['x'] + 5, point['y'] + 5, str(i+1), 
                                   color='white', fontsize=12, weight='bold',
                                   bbox=dict(boxstyle="round,pad=0.2", facecolor='blue', alpha=0.8))
        
        # Task description
        task_text = data['language_instruction']
        if len(task_text) > 100:
            task_text = task_text[:100] + "..."
        
        axes[1, 0].text(0.1, 0.5, f'Task: {task_text}', fontsize=16, 
                       transform=axes[1, 0].transAxes, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[1, 0].set_xlim(0, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].axis('off')
        axes[1, 0].set_title('Language Instruction', fontsize=18)
        
        # Action and state info
        action_info = f"Action: {data['action'][:3]}... (shape: {data['action'].shape})"
        state_info = f"State: {data['observation']['state'][:3]}... (shape: {data['observation']['state'].shape})"
        reward_info = f"Reward: {data['reward']:.4f}"
        flags_info = f"Flags: Terminal={data['is_terminal']}, Last={data['is_last']}, First={data['is_first']}"
        
        info_text = f"{action_info}\n{state_info}\n{reward_info}\n{flags_info}"
        axes[1, 1].text(0.1, 0.5, info_text, fontsize=14, 
                       transform=axes[1, 1].transAxes, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Action & State Info', fontsize=18)
        
        # 키보드 컨트롤 안내 텍스트 추가
        control_text = "Controls: ↑ (forward 10), ↓ (backward 10), n/→ (next), p/← (prev), s (save), l (label), c (DLT), e (copy to future), d (copy to past), t (clear all), r (remove), w (save episode), q (quit), h (help)"
        self.fig.text(0.5, 0.02, control_text, ha='center', va='bottom', 
                     fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # 라벨링 모드 안내 텍스트
        if self.labeling_mode:
            label_text = "LABELING MODE: Left-click to add points, Right-click to remove points. Points auto-copy between steps. Press 'l' to finish."
            self.fig.text(0.5, 0.06, label_text, ha='center', va='bottom', 
                         fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.8))
        elif self.dlt_mode:
            dlt_text = f"DLT MODE: Click on base image to collect 2D-3D correspondences ({len(self.dlt_points_2d)}/6). Press 'c' to compute transform."
            self.fig.text(0.5, 0.06, dlt_text, ha='center', va='bottom', 
                         fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.8))
        elif self.copy_to_all_steps_mode:
            copy_text = "COPY TO FUTURE STEPS MODE: Click on any labeled point to copy it to all future steps. Press 'e' again to exit."
            self.fig.text(0.5, 0.06, copy_text, ha='center', va='bottom', 
                         fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="purple", alpha=0.8))
        elif self.copy_to_past_steps_mode:
            copy_text = "COPY TO PAST STEPS MODE: Click on any labeled point to copy it to all past steps. Press 'd' again to exit."
            self.fig.text(0.5, 0.06, copy_text, ha='center', va='bottom', 
                         fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.8))
        elif self.auto_labeling_edit_mode:
            edit_text = "AUTO-LABELING EDIT MODE: Left-click to add points, Right-click to remove points. Press 'a' again to exit."
            self.fig.text(0.5, 0.06, edit_text, ha='center', va='bottom', 
                         fontsize=16, bbox=dict(boxstyle="round,pad=0.3", facecolor="cyan", alpha=0.8))
        
        plt.tight_layout()
        
        
        # Non-blocking show
        plt.show(block=False)
        plt.pause(0.1)  # 이미지가 표시될 때까지 잠시 대기
        
        # 키보드 이벤트 핸들러 다시 연결
        if hasattr(self, 'on_key_handler'):
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_handler)
        
        # 마우스 클릭 이벤트 핸들러 연결
        if hasattr(self, 'on_click_handler'):
            self.fig.canvas.mpl_connect('button_press_event', self.on_click_handler)
    
    def interactive_visualizer(self, episode_idx: int, start_step: int = 0, end_step: Optional[int] = None):
        """인터랙티브 시각화 (키보드로 네비게이션)"""
        episode_info = self.get_episode_info(episode_idx)
        num_steps = episode_info['num_steps']
        
        # start_step이 -1이면 가장 마지막 스텝으로 설정
        if start_step == -1:
            start_step = num_steps - 1
        
        if end_step is None:
            end_step = num_steps - 1
        
        self.episode_idx = episode_idx
        self.start_step = start_step
        self.end_step = end_step
        self.current_step = start_step
        
        print(f"\n=== Interactive Visualizer ===")
        print(f"Episode {episode_idx}: {episode_info['dataset_source']}")
        print(f"Steps: {start_step} to {end_step} (total: {num_steps})")
        print(f"Task: {self.get_step_data(episode_idx, start_step)['language_instruction']}")
        print("\nControls (use keyboard in the plot window):")
        print("  'up arrow': Move forward 10 steps")
        print("  'down arrow': Move backward 10 steps")
        print("  'n' or 'right arrow': Next step")
        print("  'p' or 'left arrow': Previous step")
        print("  's': Save current image")
        print("  'l': Toggle labeling mode")
        print("  'c': Toggle DLT mode (2D-3D transformation)")
        print("  'e': Toggle copy to future steps mode (click point to copy to all future steps)")
        print("  'd': Toggle copy to past steps mode (click point to copy to all past steps)")
        print("  't': Clear all points from all steps in the episode")
        print("  'r': Remove all points from current step")
        print("  'w': Save current episode with labels to new h5py file")
        print("  'q': Quit")
        print("  'h': Show this help")
        print("\nLabeling mode features:")
        print("  - Left-click: Add points")
        print("  - Right-click: Remove individual points")
        print("  - In labeling mode, points are automatically copied when navigating between steps")
        print("\nDLT mode features:")
        print("  - Click on base image to collect 2D-3D point correspondences")
        print("  - Need at least 6 points for DLT transform")
        print("  - DLT points are automatically saved as label points")
        print("  - Right-click to remove DLT points")
        print("  - Auto-labeling based on robot EE pose after transform computation")
        print("  - Press 'a' to toggle auto-labeling and edit auto-labeled points")
        print("  - DLT transform matrix is automatically updated when editing points")
        print("\nCopy to future steps mode features:")
        print("  - Press 'e' to enter copy mode")
        print("  - Click on any labeled point to copy it to all future steps")
        print("  - Press 'e' again to exit copy mode")
        print("\nCopy to past steps mode features:")
        print("  - Press 'd' to enter copy mode")
        print("  - Click on any labeled point to copy it to all past steps")
        print("  - Press 'd' again to exit copy mode")
        print("\nClear all points feature:")
        print("  - Press 't' to clear all labeled points from all steps in the episode")
        print("  - This action cannot be undone!")
        
        # 키보드 이벤트 핸들러 설정
        def on_key(event):
            if event.key == 'up':
                # 10개씩 앞으로 이동
                target_step = min(self.current_step + 10, self.end_step)
                if target_step > self.current_step:
                    print(f"\n=== Moving forward 10 steps: {self.current_step} -> {target_step} ===")
                    
                    # 라벨링 모드에서 점들 복사
                    previous_points = []
                    if self.labeling_mode and self.clicked_points:
                        previous_points = self.clicked_points.copy()
                        print(f"Copying {len(previous_points)} points from current step to target step")
                    
                    self.current_step = target_step
                    self.update_visualization()
                    
                    # 라벨링 모드에서 점들을 복사
                    if self.labeling_mode and previous_points:
                        self.clicked_points = previous_points.copy()
                        # 복사된 점들을 h5py 파일에 저장
                        success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, self.clicked_points)
                        if success:
                            print(f"Copied {len(self.clicked_points)} points to step {self.current_step}")
                        else:
                            print("Failed to save copied points to h5py file")
                        # 시각화 업데이트 (복사된 점들 표시)
                        self.update_visualization()
                    
                    # DLT 변환 행렬이 있고 자동 라벨링이 활성화된 경우에만 적용
                    if self.dlt_transform_matrix is not None and self.auto_labeling_enabled:
                        auto_points = self.auto_label_with_dlt(self.episode_idx, self.current_step)
                        if auto_points:
                            # 기존 점들과 합치기
                            existing_points = self.load_point_labels_from_h5(self.episode_idx, self.current_step)
                            all_points = existing_points + auto_points
                            
                            # 저장
                            success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, all_points)
                            if success:
                                print(f"Auto-labeled {len(auto_points)} points for step {self.current_step}")
                                self.update_visualization()
                            else:
                                print("Failed to save auto-labeled points")
                else:
                    print(f"Already at the end step ({self.current_step})")
            elif event.key == 'down':
                # 10개씩 뒤로 이동
                target_step = max(self.current_step - 10, self.start_step)
                if target_step < self.current_step:
                    print(f"\n=== Moving backward 10 steps: {self.current_step} -> {target_step} ===")
                    
                    # 라벨링 모드에서 점들 복사
                    current_points = []
                    if self.labeling_mode and self.clicked_points:
                        current_points = self.clicked_points.copy()
                        print(f"Copying {len(current_points)} points from current step to target step")
                    
                    self.current_step = target_step
                    self.update_visualization()
                    
                    # 라벨링 모드에서 점들을 복사
                    if self.labeling_mode and current_points:
                        self.clicked_points = current_points.copy()
                        # 복사된 점들을 h5py 파일에 저장
                        success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, self.clicked_points)
                        if success:
                            print(f"Copied {len(self.clicked_points)} points to step {self.current_step}")
                        else:
                            print("Failed to save copied points to h5py file")
                        # 시각화 업데이트 (복사된 점들 표시)
                        self.update_visualization()
                    
                    # DLT 변환 행렬이 있고 자동 라벨링이 활성화된 경우에만 적용
                    if self.dlt_transform_matrix is not None and self.auto_labeling_enabled:
                        auto_points = self.auto_label_with_dlt(self.episode_idx, self.current_step)
                        if auto_points:
                            # 기존 점들과 합치기
                            existing_points = self.load_point_labels_from_h5(self.episode_idx, self.current_step)
                            all_points = existing_points + auto_points
                            
                            # 저장
                            success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, all_points)
                            if success:
                                print(f"Auto-labeled {len(auto_points)} points for step {self.current_step}")
                                self.update_visualization()
                            else:
                                print("Failed to save auto-labeled points")
                else:
                    print(f"Already at the start step ({self.current_step})")
            elif event.key == 'right' or event.key == 'n':
                if self.current_step < self.end_step:
                    # 라벨링 모드에서 다음 스텝으로 넘어갈 때 이전 스텝의 점들을 복사
                    previous_points = []
                    if self.labeling_mode and self.clicked_points:
                        previous_points = self.clicked_points.copy()
                        print(f"Copying {len(previous_points)} points from previous step to next step")
                    
                    self.current_step += 1
                    self.update_visualization()
                    
                    # 라벨링 모드에서 이전 점들을 복사
                    if self.labeling_mode and previous_points:
                        self.clicked_points = previous_points.copy()
                        # 복사된 점들을 h5py 파일에 저장
                        success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, self.clicked_points)
                        if success:
                            print(f"Copied {len(self.clicked_points)} points to step {self.current_step}")
                        else:
                            print("Failed to save copied points to h5py file")
                        # 시각화 업데이트 (복사된 점들 표시)
                        self.update_visualization()
                    
                    # DLT 변환 행렬이 있고 자동 라벨링이 활성화된 경우에만 적용
                    if self.dlt_transform_matrix is not None and self.auto_labeling_enabled:
                        auto_points = self.auto_label_with_dlt(self.episode_idx, self.current_step)
                        if auto_points:
                            # 기존 점들과 합치기
                            existing_points = self.load_point_labels_from_h5(self.episode_idx, self.current_step)
                            all_points = existing_points + auto_points
                            
                            # 저장
                            success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, all_points)
                            if success:
                                print(f"Auto-labeled {len(auto_points)} points for step {self.current_step}")
                                self.update_visualization()
                            else:
                                print("Failed to save auto-labeled points")
            elif event.key == 'left' or event.key == 'p':
                if self.current_step > self.start_step:
                    # 라벨링 모드에서 이전 스텝으로 넘어갈 때 현재 스텝의 점들을 복사
                    current_points = []
                    if self.labeling_mode and self.clicked_points:
                        current_points = self.clicked_points.copy()
                        print(f"Copying {len(current_points)} points from current step to previous step")
                    
                    self.current_step -= 1
                    self.update_visualization()
                    
                    # 라벨링 모드에서 현재 점들을 복사
                    if self.labeling_mode and current_points:
                        self.clicked_points = current_points.copy()
                        # 복사된 점들을 h5py 파일에 저장
                        success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, self.clicked_points)
                        if success:
                            print(f"Copied {len(self.clicked_points)} points to step {self.current_step}")
                        else:
                            print("Failed to save copied points to h5py file")
                        # 시각화 업데이트 (복사된 점들 표시)
                        self.update_visualization()
                    
                    # DLT 변환 행렬이 있고 자동 라벨링이 활성화된 경우에만 적용
                    if self.dlt_transform_matrix is not None and self.auto_labeling_enabled:
                        auto_points = self.auto_label_with_dlt(self.episode_idx, self.current_step)
                        if auto_points:
                            # 기존 점들과 합치기
                            existing_points = self.load_point_labels_from_h5(self.episode_idx, self.current_step)
                            all_points = existing_points + auto_points
                            
                            # 저장
                            success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, all_points)
                            if success:
                                print(f"Auto-labeled {len(auto_points)} points for step {self.current_step}")
                                self.update_visualization()
                            else:
                                print("Failed to save auto-labeled points")
            elif event.key == 's':
                # 현재 라벨링된 점들을 h5py 파일에 저장
                if self.clicked_points:
                    success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, self.clicked_points)
                    if success:
                        print(f"Points saved to h5py file for episode {self.episode_idx}, step {self.current_step}")
                    else:
                        print("Failed to save points to h5py file")
                else:
                    print("No points to save")
            elif event.key == 'r':
                # 현재 스텝의 모든 라벨링된 점들 삭제
                success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, [])
                if success:
                    print(f"Removed all points from episode {self.episode_idx}, step {self.current_step}")
                    # 현재 클릭된 점들도 초기화
                    self.clicked_points = []
                    # 시각화 업데이트
                    self.update_visualization()
                else:
                    print("Failed to remove points from h5py file")
            elif event.key == 'w':
                # 현재 에피소드를 라벨링된 점들과 함께 새로운 h5py 파일로 저장
                self.save_episode_with_labels()
            elif event.key == 'l':
                # 라벨링 모드 토글
                self.labeling_mode = not self.labeling_mode
                if self.labeling_mode:
                    print(f"\n=== LABELING MODE ENABLED ===")
                    print("Left-click to add points, Right-click to remove points. Press 'l' again to finish labeling.")
                    # 기존 점들을 로드하여 편집 가능하게 함
                    existing_points = self.load_point_labels_from_h5(self.episode_idx, self.current_step)
                    self.clicked_points = existing_points.copy()
                    print(f"Loaded {len(existing_points)} existing points for editing")
                else:
                    print(f"\n=== LABELING MODE DISABLED ===")
                    # 라벨링 모드에서 점을 추가/삭제할 때마다 이미 저장했으므로 여기서는 저장하지 않음
                    print(f"Labeling completed with {len(self.clicked_points)} points")
                    self.clicked_points = []
                # 시각화 업데이트 (라벨링 모드 표시를 위해)
                self.update_visualization()
            elif event.key == 'c':
                # DLT 모드 토글
                if not self.dlt_mode:
                    # DLT 모드 시작
                    self.dlt_mode = True
                    self.dlt_points_2d = []
                    self.dlt_points_3d = []
                    print(f"\n=== DLT MODE ENABLED ===")
                    print("Click on base image (left) to collect 2D-3D point correspondences.")
                    print("Need at least 6 point correspondences for DLT transform.")
                    print("Tips for better DLT accuracy:")
                    print("  - Collect points across the entire image (not clustered)")
                    print("  - Include points at different depths (various 3D positions)")
                    print("  - Avoid points that are too close together (< 30px)")
                    print("  - Try to cover corners and edges of the workspace")
                    print("Press 'c' again to compute transform and start auto-labeling.")
                else:
                    # DLT 모드 종료 및 변환 행렬 계산
                    total_dlt_points = len(self.all_dlt_points_2d)
                    if total_dlt_points >= 6:
                        print(f"\n=== Computing Transform ===")
                        print(f"Collected {total_dlt_points} total DLT point correspondences")
                        print(f"Current session points: {len(self.dlt_points_2d)}")
                        print(f"Previous session points: {total_dlt_points - len(self.dlt_points_2d)}")
                        
                        # 변환 행렬 계산 (모든 DLT 점들 사용)
                        self.dlt_transform_matrix = self.compute_dlt_transform(
                            self.all_dlt_points_2d, self.all_dlt_points_3d
                        )
                        
                        if self.dlt_transform_matrix is not None:
                            print("DLT transform matrix computed successfully!")
                            print("Press 'a' to toggle auto-labeling on/off.")
                            
                            # 자동 라벨링을 기본적으로 비활성화 상태로 유지
                            self.auto_labeling_enabled = False
                        else:
                            print("Failed to compute DLT transform matrix")
                    else:
                        print(f"Not enough total point correspondences ({total_dlt_points} < 6)")
                        print(f"Current session points: {len(self.dlt_points_2d)}")
                        print(f"Previous session points: {total_dlt_points - len(self.dlt_points_2d)}")
                    
                    self.dlt_mode = False
                    self.dlt_points_2d = []
                    self.dlt_points_3d = []
                    print("=== DLT MODE DISABLED ===")
                
                # 시각화 업데이트
                self.update_visualization()
            elif event.key == 'e':
                # 미래 스텝에 복사 모드 토글
                self.copy_to_all_steps_mode = not self.copy_to_all_steps_mode
                if self.copy_to_all_steps_mode:
                    print(f"\n=== COPY TO FUTURE STEPS MODE ENABLED ===")
                    print("Click on any labeled point to copy it to all future steps.")
                    print("Press 'e' again to exit copy mode.")
                else:
                    print(f"\n=== COPY TO FUTURE STEPS MODE DISABLED ===")
                    print("Copy mode is now disabled.")
                
                # 시각화 업데이트
                self.update_visualization()
            elif event.key == 'd':
                # 이전 스텝에 복사 모드 토글
                self.copy_to_past_steps_mode = not self.copy_to_past_steps_mode
                if self.copy_to_past_steps_mode:
                    print(f"\n=== COPY TO PAST STEPS MODE ENABLED ===")
                    print("Click on any labeled point to copy it to all past steps.")
                    print("Press 'd' again to exit copy mode.")
                else:
                    print(f"\n=== COPY TO PAST STEPS MODE DISABLED ===")
                    print("Copy mode is now disabled.")
                
                # 시각화 업데이트
                self.update_visualization()
            elif event.key == 't':
                # 모든 스텝의 모든 점 삭제
                print(f"\n=== Clearing All Points from All Steps ===")
                print("This will remove ALL labeled points from ALL steps in the episode.")
                print("This action cannot be undone!")
                
                # 사용자 확인
                confirm = input("Are you sure? Type 'yes' to confirm: ")
                if confirm.lower() == 'yes':
                    episode_info = self.get_episode_info(self.episode_idx)
                    num_steps = episode_info['num_steps']
                    
                    cleared_count = 0
                    for step_idx in range(num_steps):
                        # 각 스텝의 점들 삭제
                        success = self.save_point_labels_to_h5(self.episode_idx, step_idx, [])
                        if success:
                            cleared_count += 1
                        else:
                            print(f"Failed to clear points from step {step_idx}")
                    
                    print(f"Successfully cleared all points from {cleared_count}/{num_steps} steps")
                    
                    # 현재 클릭된 점들도 초기화
                    self.clicked_points = []
                    
                    # 시각화 업데이트
                    self.update_visualization()
                else:
                    print("Clear operation cancelled.")
            elif event.key == 'a':
                # 자동 라벨링 토글
                if self.dlt_transform_matrix is not None:
                    if not self.auto_labeling_enabled:
                        # 자동 라벨링 활성화
                        self.auto_labeling_enabled = True
                        self.auto_labeling_edit_mode = False
                        print(f"\n=== AUTO-LABELING ENABLED ===")
                        print("Auto-labeling will be applied when navigating between steps.")
                        print("Press 'a' again to enter edit mode for auto-labeled points.")
                        
                        # 현재 스텝에 자동 라벨링 적용
                        auto_points = self.auto_label_with_dlt(self.episode_idx, self.current_step)
                        if auto_points:
                            # 기존 점들과 합치기
                            existing_points = self.load_point_labels_from_h5(self.episode_idx, self.current_step)
                            all_points = existing_points + auto_points
                            
                            # 저장
                            success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, all_points)
                            if success:
                                print(f"Auto-labeled {len(auto_points)} points for current step")
                                self.update_visualization()
                            else:
                                print("Failed to save auto-labeled points")
                        else:
                            print("No auto-labeled points generated for current step")
                    else:
                        # 자동 라벨링 편집 모드 토글
                        if not self.auto_labeling_edit_mode:
                            self.auto_labeling_edit_mode = True
                            print(f"\n=== AUTO-LABELING EDIT MODE ENABLED ===")
                            print("You can now edit auto-labeled points.")
                            print("Left-click to add points, Right-click to remove points.")
                            print("DLT transform matrix will be updated automatically when you modify points.")
                            print("Press 'a' again to disable auto-labeling completely.")
                        else:
                            # 자동 라벨링 완전 비활성화
                            self.auto_labeling_enabled = False
                            self.auto_labeling_edit_mode = False
                            print(f"\n=== AUTO-LABELING DISABLED ===")
                            print("Auto-labeling is now completely disabled.")
                else:
                    print("No DLT transform matrix available. Please compute transform first with 'c' key.")
                
                # 시각화 업데이트
                self.update_visualization()
            elif event.key == 'q':
                self.running = False
                plt.close('all')
            elif event.key == 'h':
                print("\nControls:")
                print("  'up arrow': Move forward 10 steps")
                print("  'down arrow': Move backward 10 steps")
                print("  'n' or 'right arrow': Next step")
                print("  'p' or 'left arrow': Previous step")
                print("  's': Save current points to h5py file")
                print("  'l': Toggle labeling mode (click to add points)")
                print("  'c': Toggle DLT mode (2D-3D transformation)")
                print("  'e': Toggle copy to future steps mode (click point to copy to all future steps)")
                print("  'd': Toggle copy to past steps mode (click point to copy to all past steps)")
                print("  't': Clear all points from all steps in the episode")
                print("  'r': Remove all points from current step")
                print("  'w': Save current episode with labels to new h5py file")
                print("  'q': Quit")
                print("  'h': Show this help")
                print("\nLabeling mode features:")
                print("  - Left-click: Add points")
                print("  - Right-click: Remove individual points")
                print("  - In labeling mode, points are automatically copied when navigating between steps")
                print("\nDLT mode features:")
                print("  - Click on base image to collect 2D-3D point correspondences")
                print("  - Need at least 6 points for DLT transform")
                print("  - Auto-labeling based on robot EE pose after transform computation")
                print("\nCopy to future steps mode features:")
                print("  - Press 'e' to enter copy mode")
                print("  - Click on any labeled point to copy it to all future steps")
                print("  - Press 'e' again to exit copy mode")
                print("\nClear all points feature:")
                print("  - Press 't' to clear all labeled points from all steps in the episode")
                print("  - This action cannot be undone!")
        
        # 마우스 클릭 이벤트 핸들러 설정 (좌클릭: 점 추가, 우클릭: 점 삭제)
        def on_click(event):
            if not (self.labeling_mode or self.dlt_mode or self.copy_to_all_steps_mode or self.copy_to_past_steps_mode or self.auto_labeling_edit_mode):
                return
            
            # 클릭한 위치가 이미지 영역인지 확인
            if event.inaxes in [self.axes[0, 0], self.axes[0, 1]]:
                # 이미지 좌표로 변환
                x, y = event.xdata, event.ydata
                if x is not None and y is not None:
                    # 어떤 이미지인지 구분
                    image_type = "base" if event.inaxes == self.axes[0, 0] else "wrist"
                    
                    # DLT 모드에서의 동작 (base 이미지만 사용)
                    if self.dlt_mode:
                        if event.button == 1 and image_type == "base":  # base 이미지에서만 좌클릭 처리
                            # 2D 점 추가
                            self.dlt_points_2d.append([x, y])
                            
                            # 현재 스텝의 로봇 EE pose 가져오기
                            ee_pose = self.get_robot_ee_pose(self.episode_idx, self.current_step)
                            if ee_pose is not None:
                                ee_position = ee_pose[:3]  # x, y, z
                                self.dlt_points_3d.append(ee_position)
                                
                                # 전체 에피소드 DLT 점들에도 추가
                                self.all_dlt_points_2d.append([x, y])
                                self.all_dlt_points_3d.append(ee_position)
                                self.all_dlt_steps.append(self.current_step)
                                
                                print(f"DLT point {len(self.dlt_points_2d)}: 2D({x:.1f}, {y:.1f}) -> 3D({ee_position[0]:.3f}, {ee_position[1]:.3f}, {ee_position[2]:.3f})")
                                print(f"Total DLT points collected: {len(self.all_dlt_points_2d)}")
                                
                                # DLT 점을 실제 라벨링으로도 저장
                                dlt_point = {
                                    'x': float(x),
                                    'y': float(y),
                                    'image_type': image_type,
                                    'step': self.current_step,
                                    'dlt_point': True  # DLT 점임을 표시
                                }
                                
                                # 현재 스텝의 기존 점들 로드
                                existing_points = self.load_point_labels_from_h5(self.episode_idx, self.current_step)
                                
                                # 같은 위치에 이미 점이 있는지 확인
                                point_exists = False
                                for existing_point in existing_points:
                                    if (existing_point['image_type'] == image_type and
                                        abs(existing_point['x'] - x) < 5 and
                                        abs(existing_point['y'] - y) < 5):
                                        point_exists = True
                                        break
                                
                                if not point_exists:
                                    existing_points.append(dlt_point)
                                    # h5py 파일에 저장
                                    success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, existing_points)
                                    if success:
                                        print(f"DLT point also saved as label point")
                                    else:
                                        print("Failed to save DLT point as label")
                                
                                # 점을 이미지에 표시 (DLT용)
                                event.inaxes.plot(x, y, 's', color='green', markersize=12, markeredgecolor='white', markeredgewidth=2)
                                event.inaxes.text(x + 5, y + 5, f"D{len(self.dlt_points_2d)}", 
                                                color='white', fontsize=12, weight='bold',
                                                bbox=dict(boxstyle="round,pad=0.2", facecolor='green', alpha=0.8))
                                
                                # 캔버스 업데이트
                                self.fig.canvas.draw()
                            else:
                                print("Could not get robot EE pose for DLT point")
                                # 2D 점 제거
                                self.dlt_points_2d.pop()
                        elif event.button == 1 and image_type == "wrist":
                            print("DLT mode: Please click on base image (left) only, not wrist image (right)")
                        elif event.button == 3:  # 우클릭으로 DLT 점 삭제
                            # 가장 가까운 DLT 점 찾기
                            min_distance = float('inf')
                            closest_dlt_idx = -1
                            
                            for i, (point_2d, point_3d) in enumerate(zip(self.dlt_points_2d, self.dlt_points_3d)):
                                distance = ((point_2d[0] - x) ** 2 + (point_2d[1] - y) ** 2) ** 0.5
                                if distance < min_distance and distance < 20:  # 20픽셀 이내
                                    min_distance = distance
                                    closest_dlt_idx = i
                            
                            if closest_dlt_idx >= 0:
                                # DLT 점 삭제
                                removed_2d = self.dlt_points_2d.pop(closest_dlt_idx)
                                removed_3d = self.dlt_points_3d.pop(closest_dlt_idx)
                                
                                # 전체 에피소드 DLT 점들에서도 삭제 (가장 가까운 점 찾기)
                                min_distance_all = float('inf')
                                closest_all_idx = -1
                                for i, (point_2d, point_3d) in enumerate(zip(self.all_dlt_points_2d, self.all_dlt_points_3d)):
                                    distance = ((point_2d[0] - removed_2d[0]) ** 2 + (point_2d[1] - removed_2d[1]) ** 2) ** 0.5
                                    if distance < min_distance_all and distance < 5:  # 5픽셀 이내
                                        min_distance_all = distance
                                        closest_all_idx = i
                                
                                if closest_all_idx >= 0:
                                    self.all_dlt_points_2d.pop(closest_all_idx)
                                    self.all_dlt_points_3d.pop(closest_all_idx)
                                    self.all_dlt_steps.pop(closest_all_idx)
                                    print(f"Removed DLT point {closest_dlt_idx + 1}: 2D({removed_2d[0]:.1f}, {removed_2d[1]:.1f})")
                                    print(f"Total DLT points remaining: {len(self.all_dlt_points_2d)}")
                                else:
                                    print(f"Removed DLT point {closest_dlt_idx + 1}: 2D({removed_2d[0]:.1f}, {removed_2d[1]:.1f}) (not found in all points)")
                                
                                # 해당 점을 라벨링에서도 삭제
                                existing_points = self.load_point_labels_from_h5(self.episode_idx, self.current_step)
                                points_to_remove = []
                                for i, point in enumerate(existing_points):
                                    if (point['image_type'] == image_type and
                                        abs(point['x'] - removed_2d[0]) < 5 and
                                        abs(point['y'] - removed_2d[1]) < 5):
                                        points_to_remove.append(i)
                                
                                # 뒤에서부터 삭제 (인덱스 변화 방지)
                                for i in reversed(points_to_remove):
                                    existing_points.pop(i)
                                
                                # 저장
                                success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, existing_points)
                                if success:
                                    print(f"Removed corresponding label point")
                                
                                # 시각화 업데이트
                                self.update_visualization()
                            else:
                                print("No DLT point found near the clicked location")
                        return
                    
                    # 복사 모드에서의 동작
                    if self.copy_to_all_steps_mode:
                        if event.button == 1:  # 좌클릭
                            # 현재 스텝의 모든 점들 확인
                            current_points = self.load_point_labels_from_h5(self.episode_idx, self.current_step)
                            
                            # 클릭한 위치에서 가장 가까운 점 찾기
                            min_distance = float('inf')
                            closest_point = None
                            
                            for point in current_points:
                                if point['image_type'] == image_type:
                                    distance = ((point['x'] - x) ** 2 + (point['y'] - y) ** 2) ** 0.5
                                    if distance < min_distance and distance < 20:  # 20픽셀 이내
                                        min_distance = distance
                                        closest_point = point.copy()
                            
                            if closest_point is not None:
                                print(f"\n=== Copying Point to Future Steps ===")
                                print(f"Selected point: ({closest_point['x']:.1f}, {closest_point['y']:.1f}) on {closest_point['image_type']} image")
                                
                                # 에피소드의 미래 스텝들에만 복사 (현재 스텝 이후)
                                episode_info = self.get_episode_info(self.episode_idx)
                                num_steps = episode_info['num_steps']
                                
                                copied_count = 0
                                for step_idx in range(self.current_step + 1, num_steps):
                                    # 현재 스텝의 점들 로드
                                    existing_points = self.load_point_labels_from_h5(self.episode_idx, step_idx)
                                    
                                    # 같은 위치에 이미 점이 있는지 확인
                                    point_exists = False
                                    for existing_point in existing_points:
                                        if (existing_point['image_type'] == closest_point['image_type'] and
                                            abs(existing_point['x'] - closest_point['x']) < 5 and
                                            abs(existing_point['y'] - closest_point['y']) < 5):
                                            point_exists = True
                                            break
                                    
                                    if not point_exists:
                                        # 점 추가
                                        new_point = closest_point.copy()
                                        new_point['step'] = step_idx
                                        existing_points.append(new_point)
                                        
                                        # 저장
                                        success = self.save_point_labels_to_h5(self.episode_idx, step_idx, existing_points)
                                        if success:
                                            copied_count += 1
                                        else:
                                            print(f"Failed to save point to step {step_idx}")
                                
                                print(f"Successfully copied point to {copied_count}/{num_steps - self.current_step - 1} future steps")
                                
                                # 복사 모드 종료
                                self.copy_to_all_steps_mode = False
                                print("Copy mode disabled. Press 'e' to enter copy mode again.")
                                
                                # 시각화 업데이트
                                self.update_visualization()
                            else:
                                print("No point found near the clicked location")
                        return
                    
                    # 이전 스텝 복사 모드에서의 동작
                    if self.copy_to_past_steps_mode:
                        if event.button == 1:  # 좌클릭
                            # 현재 스텝의 모든 점들 확인
                            current_points = self.load_point_labels_from_h5(self.episode_idx, self.current_step)
                            
                            # 클릭한 위치에서 가장 가까운 점 찾기
                            min_distance = float('inf')
                            closest_point = None
                            
                            for point in current_points:
                                if point['image_type'] == image_type:
                                    distance = ((point['x'] - x) ** 2 + (point['y'] - y) ** 2) ** 0.5
                                    if distance < min_distance and distance < 20:  # 20픽셀 이내
                                        min_distance = distance
                                        closest_point = point.copy()
                            
                            if closest_point is not None:
                                print(f"\n=== Copying Point to Past Steps ===")
                                print(f"Selected point: ({closest_point['x']:.1f}, {closest_point['y']:.1f}) on {closest_point['image_type']} image")
                                
                                # 에피소드의 이전 스텝들에만 복사 (현재 스텝 이전)
                                episode_info = self.get_episode_info(self.episode_idx)
                                num_steps = episode_info['num_steps']
                                
                                copied_count = 0
                                for step_idx in range(self.current_step):
                                    # 현재 스텝의 점들 로드
                                    existing_points = self.load_point_labels_from_h5(self.episode_idx, step_idx)
                                    
                                    # 같은 위치에 이미 점이 있는지 확인
                                    point_exists = False
                                    for existing_point in existing_points:
                                        if (existing_point['image_type'] == closest_point['image_type'] and
                                            abs(existing_point['x'] - closest_point['x']) < 5 and
                                            abs(existing_point['y'] - closest_point['y']) < 5):
                                            point_exists = True
                                            break
                                    
                                    if not point_exists:
                                        # 점 추가
                                        new_point = closest_point.copy()
                                        new_point['step'] = step_idx
                                        existing_points.append(new_point)
                                        
                                        # 저장
                                        success = self.save_point_labels_to_h5(self.episode_idx, step_idx, existing_points)
                                        if success:
                                            copied_count += 1
                                        else:
                                            print(f"Failed to save point to step {step_idx}")
                                
                                print(f"Successfully copied point to {copied_count}/{self.current_step} past steps")
                                
                                # 복사 모드 종료
                                self.copy_to_past_steps_mode = False
                                print("Copy mode disabled. Press 'd' to enter copy mode again.")
                                
                                # 시각화 업데이트
                                self.update_visualization()
                            else:
                                print("No point found near the clicked location")
                        return
                    
                    # 자동 라벨링 편집 모드에서의 동작
                    if self.auto_labeling_edit_mode:
                        # 우클릭인지 확인 (점 삭제)
                        if event.button == 3:  # 우클릭
                            # 현재 스텝의 모든 점들 확인
                            current_points = self.load_point_labels_from_h5(self.episode_idx, self.current_step)
                            
                            # 가장 가까운 점 찾기
                            min_distance = float('inf')
                            closest_point_idx = -1
                            
                            for i, point in enumerate(current_points):
                                if point['image_type'] == image_type:
                                    distance = ((point['x'] - x) ** 2 + (point['y'] - y) ** 2) ** 0.5
                                    if distance < min_distance and distance < 20:  # 20픽셀 이내
                                        min_distance = distance
                                        closest_point_idx = i
                            
                            # 가장 가까운 점 삭제
                            if closest_point_idx >= 0:
                                removed_point = current_points.pop(closest_point_idx)
                                print(f"Removed point at ({removed_point['x']:.1f}, {removed_point['y']:.1f}) from {image_type} image")
                                
                                # base 이미지의 점이면 DLT 점들에서도 삭제
                                if image_type == "base":
                                    # 가장 가까운 DLT 점 찾기
                                    min_distance_dlt = float('inf')
                                    closest_dlt_idx = -1
                                    for i, (point_2d, point_3d) in enumerate(zip(self.all_dlt_points_2d, self.all_dlt_points_3d)):
                                        distance = ((point_2d[0] - removed_point['x']) ** 2 + (point_2d[1] - removed_point['y']) ** 2) ** 0.5
                                        if distance < min_distance_dlt and distance < 5:  # 5픽셀 이내
                                            min_distance_dlt = distance
                                            closest_dlt_idx = i
                                    
                                    if closest_dlt_idx >= 0:
                                        self.all_dlt_points_2d.pop(closest_dlt_idx)
                                        self.all_dlt_points_3d.pop(closest_dlt_idx)
                                        self.all_dlt_steps.pop(closest_dlt_idx)
                                        print(f"Removed corresponding DLT point")
                                        print(f"Total DLT points remaining: {len(self.all_dlt_points_2d)}")
                                    else:
                                        print(f"Corresponding DLT point not found")
                                
                                # 저장
                                success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, current_points)
                                if success:
                                    print(f"Updated h5py file with {len(current_points)} points")
                                else:
                                    print("Failed to update h5py file")
                                
                                # DLT 변환 행렬 업데이트 시도
                                self.update_dlt_transform_from_current_points()
                                
                                # 시각화 업데이트
                                self.update_visualization()
                            else:
                                print("No point found near the clicked location")
                        
                        # 좌클릭인지 확인 (점 추가)
                        elif event.button == 1:  # 좌클릭
                            # 현재 스텝의 점들 로드
                            current_points = self.load_point_labels_from_h5(self.episode_idx, self.current_step)
                            
                            # 점 추가
                            point = {
                                'x': float(x),
                                'y': float(y),
                                'image_type': image_type,
                                'step': self.current_step,
                                'manual_edit': True  # 수동 편집으로 추가된 점임을 표시
                            }
                            current_points.append(point)
                            
                            # base 이미지에 추가된 점이면 DLT 점으로도 추가
                            if image_type == "base":
                                # 현재 스텝의 로봇 EE pose 가져오기
                                ee_pose = self.get_robot_ee_pose(self.episode_idx, self.current_step)
                                if ee_pose is not None:
                                    ee_position = ee_pose[:3]  # x, y, z
                                    
                                    # 중복 확인
                                    is_duplicate = False
                                    for existing_2d in self.all_dlt_points_2d:
                                        distance = ((x - existing_2d[0]) ** 2 + (y - existing_2d[1]) ** 2) ** 0.5
                                        if distance < 5:  # 5픽셀 이내면 중복으로 간주
                                            is_duplicate = True
                                            break
                                    
                                    if not is_duplicate:
                                        self.all_dlt_points_2d.append([x, y])
                                        self.all_dlt_points_3d.append(ee_position)
                                        self.all_dlt_steps.append(self.current_step)
                                        print(f"Added point to DLT collection: 2D({x:.1f}, {y:.1f}) -> 3D({ee_position[0]:.3f}, {ee_position[1]:.3f}, {ee_position[2]:.3f})")
                                        print(f"Total DLT points: {len(self.all_dlt_points_2d)}")
                                    else:
                                        print(f"Point at ({x:.1f}, {y:.1f}) already exists in DLT collection")
                                else:
                                    print(f"Could not get robot EE pose for DLT point at ({x:.1f}, {y:.1f})")
                            
                            # 저장
                            success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, current_points)
                            if success:
                                print(f"Added manual point at ({x:.1f}, {y:.1f}) on {image_type} image")
                                print(f"Updated h5py file with {len(current_points)} points")
                            else:
                                print("Failed to update h5py file")
                            
                            # DLT 변환 행렬 업데이트 시도
                            self.update_dlt_transform_from_current_points()
                            
                            # 시각화 업데이트
                            self.update_visualization()
                        return
                    
                    # 일반 라벨링 모드에서의 동작
                    # 우클릭인지 확인 (점 삭제)
                    if event.button == 3:  # 우클릭
                        # 가장 가까운 점 찾기
                        min_distance = float('inf')
                        closest_point_idx = -1
                        
                        for i, point in enumerate(self.clicked_points):
                            if point['image_type'] == image_type:
                                distance = ((point['x'] - x) ** 2 + (point['y'] - y) ** 2) ** 0.5
                                if distance < min_distance and distance < 20:  # 20픽셀 이내
                                    min_distance = distance
                                    closest_point_idx = i
                        
                        # 가장 가까운 점 삭제
                        if closest_point_idx >= 0:
                            removed_point = self.clicked_points.pop(closest_point_idx)
                            print(f"Removed point at ({removed_point['x']:.1f}, {removed_point['y']:.1f}) from {image_type} image")
                            
                            # 라벨링 모드에서 점을 삭제한 경우 즉시 h5py 파일에 저장
                            if self.labeling_mode:
                                success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, self.clicked_points)
                                if success:
                                    print(f"Updated h5py file with {len(self.clicked_points)} points")
                                else:
                                    print("Failed to update h5py file")
                            
                            # 시각화 업데이트 (점들 다시 그리기)
                            self.update_visualization()
                        else:
                            print("No point found near the clicked location")
                    
                    # 좌클릭인지 확인 (점 추가)
                    elif event.button == 1:  # 좌클릭
                        # 점 추가
                        point = {
                            'x': float(x),
                            'y': float(y),
                            'image_type': image_type,
                            'step': self.current_step
                        }
                        self.clicked_points.append(point)
                        
                        # 점을 이미지에 표시
                        color = 'red' if image_type == "base" else 'blue'
                        event.inaxes.plot(x, y, 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=2)
                        event.inaxes.text(x + 5, y + 5, str(len(self.clicked_points)), 
                                        color='white', fontsize=12, weight='bold',
                                        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8))
                        
                        # 캔버스 업데이트
                        self.fig.canvas.draw()
                        
                        print(f"Added point {len(self.clicked_points)} at ({x:.1f}, {y:.1f}) on {image_type} image")
                        
                        # 라벨링 모드에서 점을 추가한 경우 즉시 h5py 파일에 저장
                        if self.labeling_mode:
                            success = self.save_point_labels_to_h5(self.episode_idx, self.current_step, self.clicked_points)
                            if success:
                                print(f"Updated h5py file with {len(self.clicked_points)} points")
                            else:
                                print("Failed to update h5py file")
        
        # 핸들러를 인스턴스 변수로 저장
        self.on_key_handler = on_key
        self.on_click_handler = on_click
        
        # 첫 번째 이미지 표시
        self.update_visualization()
        
        # 키보드 이벤트 연결
        if self.fig is not None:
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_handler)
            self.fig.canvas.mpl_connect('button_press_event', self.on_click_handler)
        
        # 메인 루프
        while self.running:
            try:
                plt.pause(0.1)  # GUI 이벤트 처리
                if not plt.fignum_exists(self.fig.number):
                    break
            except KeyboardInterrupt:
                print("\nInterrupted by user.")
                break
            except Exception as e:
                print(f"Error: {e}")
                break
        
        # 라벨링 결과 저장
        if self.labels:
            self.save_labels()
    
    def update_visualization(self):
        """현재 스텝의 시각화를 업데이트"""
        try:
            data = self.get_step_data(self.episode_idx, self.current_step)
            task = data['language_instruction']
            
            print(f"\n--- Step {self.current_step} ---")
            print(f"Task: {task}")
            print(f"Action shape: {data['action'].shape}")
            print(f"State shape: {data['observation']['state'].shape}")
            print(f"Reward: {data['reward']:.4f}")
            
            # 이미지 시각화
            self.visualize_step(self.episode_idx, self.current_step)
            
        except Exception as e:
            print(f"Error at step {self.current_step}: {e}")
    

    
    def save_labels(self, output_file: str = "labels.json"):
        """라벨링 결과를 JSON 파일로 저장"""
        import json
        
        with open(output_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
        print(f"Labels saved to: {output_file}")
    
    def save_point_labels_to_h5(self, episode_idx: int, step_idx: int, points: list):
        """라벨링된 점들을 h5py 파일의 observation에 저장"""
        try:
            episode_name = self.episode_names[episode_idx]
            episode_group = self.episodes_group[episode_name]
            step_name = f'step_{step_idx:06d}'
            
            if step_name not in episode_group:
                print(f"Error: Step {step_idx} not found in episode {episode_idx}")
                return False
            
            step_group = episode_group[step_name]
            observation_group = step_group['observation']
            
            # 원본 이미지 크기 가져오기
            base_image = observation_group['image'][:]
            wrist_image = observation_group['wrist_image'][:]
            base_height, base_width = base_image.shape[:2]
            wrist_height, wrist_width = wrist_image.shape[:2]
            
            # 점들을 numpy 배열로 변환
            if points:
                # 각 점을 [x, y, image_type_code] 형태로 저장
                # image_type_code: 0=base, 1=wrist
                point_array = []
                for point in points:
                    image_type_code = 0 if point['image_type'] == 'base' else 1
                    point_array.append([point['x'], point['y'], image_type_code])
                
                point_array = np.array(point_array, dtype=np.float32)
            else:
                # 빈 배열
                point_array = np.array([], dtype=np.float32).reshape(0, 3)
            
            # 기존 point_label이 있으면 삭제
            if 'point_label' in observation_group:
                del observation_group['point_label']
            
            # 새로운 point_label 데이터셋 생성
            observation_group.create_dataset('point_label', data=point_array)
            
            # point_image 생성 (라벨링된 점들의 마스크)
            # base 이미지용 마스크
            base_point_image = np.zeros((base_height, base_width), dtype=np.uint8)
            # wrist 이미지용 마스크  
            wrist_point_image = np.zeros((wrist_height, wrist_width), dtype=np.uint8)
            
            # 라벨링된 점들을 마스크에 표시
            for point in points:
                x, y = int(point['x']), int(point['y'])
                if point['image_type'] == 'base':
                    if 0 <= x < base_width and 0 <= y < base_height:
                        base_point_image[y, x] = 255  # 흰색으로 표시
                elif point['image_type'] == 'wrist':
                    if 0 <= x < wrist_width and 0 <= y < wrist_height:
                        wrist_point_image[y, x] = 255  # 흰색으로 표시
            
            # 기존 point_image가 있으면 삭제
            if 'point_image' in observation_group:
                del observation_group['point_image']
            if 'point_image_wrist' in observation_group:
                del observation_group['point_image_wrist']
            
            # 새로운 point_image 데이터셋 생성
            observation_group.create_dataset('point_image', data=base_point_image)
            observation_group.create_dataset('point_image_wrist', data=wrist_point_image)
            
            print(f"Saved {len(points)} points to h5py file: episode {episode_idx}, step {step_idx}")
            print(f"Created point masks: base({base_width}x{base_height}), wrist({wrist_width}x{wrist_height})")
            return True
            
        except Exception as e:
            print(f"Error saving points to h5py: {e}")
            return False
    
    def load_point_labels_from_h5(self, episode_idx: int, step_idx: int):
        """h5py 파일에서 라벨링된 점들을 로드"""
        try:
            episode_name = self.episode_names[episode_idx]
            episode_group = self.episodes_group[episode_name]
            step_name = f'step_{step_idx:06d}'
            
            if step_name not in episode_group:
                return []
            
            step_group = episode_group[step_name]
            observation_group = step_group['observation']
            
            if 'point_label' not in observation_group:
                return []
            
            point_array = observation_group['point_label'][:]
            points = []
            
            for i, point_data in enumerate(point_array):
                x, y, image_type_code = point_data
                image_type = 'base' if image_type_code == 0 else 'wrist'
                points.append({
                    'x': float(x),
                    'y': float(y),
                    'image_type': image_type,
                    'step': step_idx
                })
            
            return points
            
        except Exception as e:
            print(f"Error loading points from h5py: {e}")
            return []
    
    def load_point_images_from_h5(self, episode_idx: int, step_idx: int):
        """h5py 파일에서 point_image 마스크들을 로드"""
        try:
            episode_name = self.episode_names[episode_idx]
            episode_group = self.episodes_group[episode_name]
            step_name = f'step_{step_idx:06d}'
            
            if step_name not in episode_group:
                return None, None
            
            step_group = episode_group[step_name]
            observation_group = step_group['observation']
            
            base_point_image = None
            wrist_point_image = None
            
            if 'point_image' in observation_group:
                base_point_image = observation_group['point_image'][:]
            
            if 'point_image_wrist' in observation_group:
                wrist_point_image = observation_group['point_image_wrist'][:]
            
            return base_point_image, wrist_point_image
            
        except Exception as e:
            print(f"Error loading point images from h5py: {e}")
            return None, None
    
    def get_robot_ee_pose(self, episode_idx: int, step_idx: int):
        """로봇 end-effector pose를 가져옴 (state에서 첫 7개 값: [x, y, z, qw, qx, qy, qz])"""
        try:
            data = self.get_step_data(episode_idx, step_idx)
            state = data['observation']['state']
            # 처음 7개 값이 end-effector pose (position + quaternion)
            ee_pose = state[:7]
            return ee_pose
        except Exception as e:
            print(f"Error getting robot EE pose: {e}")
            return None
    
    def dlt_estimate(self, X, u):
        """DLT 알고리즘으로 3x4 프로젝션 행렬 추정"""
        print(f"DLT input shapes: X={X.shape}, u={u.shape}")
        print(f"3D points (X):\n{X}")
        print(f"2D points (u):\n{u}")
        
        A = []
        for i in range(X.shape[1]):
            x = X[:, i]
            ux, uy, _ = u[:, i]
            A.append([0, 0, 0, 0, -ux * x[0], -ux * x[1], -ux * x[2], -ux * x[3], uy * x[0], uy * x[1], uy * x[2], uy * x[3]])
            A.append([ux * x[0], ux * x[1], ux * x[2], ux * x[3], 0, 0, 0, 0, -uy * x[0], -uy * x[1], -uy * x[2], -uy * x[3]])
        A = np.array(A)
        print(f"DLT matrix A shape: {A.shape}")
        
        # SVD로 해결
        U, S, Vt = np.linalg.svd(A)
        print(f"SVD singular values: {S}")
        
        # 가장 작은 특이값에 해당하는 벡터가 해
        P = Vt[-1, :].reshape(3, 4)
        print(f"Raw DLT matrix P:\n{P}")
        
        # 정규화 (P[2,3] = 1로)
        if abs(P[2, 3]) > 1e-10:
            P = P / P[2, 3]
            print(f"Normalized DLT matrix P:\n{P}")
        else:
            print(f"Warning: P[2,3] too small ({P[2, 3]}), not normalizing")
        
        return P

    def compute_dlt_transform(self, points_2d, points_3d):
        """개선된 DLT 알고리즘을 사용하여 2D-3D 변환 행렬 계산 (3x4)"""
        if len(points_2d) < 6 or len(points_3d) < 6:
            print("Need at least 6 point correspondences for DLT")
            return None
        
        print(f"\n=== Computing Improved DLT Transform ===")
        print(f"Number of correspondences: {len(points_2d)}")
        
        # 입력 데이터 출력
        for i, (point_2d, point_3d) in enumerate(zip(points_2d, points_3d)):
            print(f"Point {i+1}: 2D({point_2d[0]:.1f}, {point_2d[1]:.1f}) -> 3D({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})")
        
        # 실제 이미지 크기 가져오기
        try:
            data = self.get_step_data(self.episode_idx, self.current_step)
            base_image = data['observation']['image']
            img_height, img_width = base_image.shape[:2]
            print(f"Actual image size: {img_width}x{img_height}")
        except:
            img_width, img_height = 320, 240
            print(f"Using default image size: {img_width}x{img_height}")
        
        # 개선된 카메라 내부 파라미터 추정
        fx = fy = max(img_width, img_height) * 1.2  # 초점 거리를 약간 증가
        cx, cy = img_width / 2, img_height / 2  # 주점
        
        # 카메라 왜곡 계수 (왜곡이 없다고 가정)
        dist_coeffs = np.zeros(5, dtype=np.float32)
        
        # DLT 점들의 품질 검사
        if len(points_2d) >= 8:
            # 점들 간의 거리 분포 확인
            distances = []
            for i in range(len(points_2d)):
                for j in range(i+1, len(points_2d)):
                    dist_2d = np.sqrt((points_2d[i][0] - points_2d[j][0])**2 + (points_2d[i][1] - points_2d[j][1])**2)
                    dist_3d = np.sqrt(sum((points_3d[i][k] - points_3d[j][k])**2 for k in range(3)))
                    distances.append((dist_2d, dist_3d))
            
            if distances:
                avg_2d_dist = sum(d[0] for d in distances) / len(distances)
                avg_3d_dist = sum(d[1] for d in distances) / len(distances)
                print(f"Point distribution - Avg 2D distance: {avg_2d_dist:.1f}px, Avg 3D distance: {avg_3d_dist:.3f}m")
                
                # 점들이 너무 가깝게 모여있으면 경고
                if avg_2d_dist < 30:
                    print("Warning: Points are too close together, DLT accuracy may be poor")
                elif avg_2d_dist > 200:
                    print("Warning: Points are too far apart, may cause numerical instability")
        
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        print(f"Improved camera matrix:\n{camera_matrix}")
        
        # RANSAC을 이용한 이상치 제거
        points_3d_cv = np.array(points_3d, dtype=np.float32)
        points_2d_cv = np.array(points_2d, dtype=np.float32)
        
        try:
            # RANSAC을 사용한 solvePnP (이상치 제거)
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d_cv, points_2d_cv, camera_matrix, dist_coeffs,
                reprojectionError=8.0,  # 재투영 오차 임계값
                confidence=0.99,  # 신뢰도
                iterationsCount=1000,  # 반복 횟수
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success and len(inliers) >= 6:
                print(f"RANSAC solvePnP successful with {len(inliers)} inliers out of {len(points_2d)}")
                print(f"Rotation vector: {rvec.flatten()}")
                print(f"Translation vector: {tvec.flatten()}")
                
                # 회전 벡터를 회전 행렬로 변환
                R, _ = cv2.Rodrigues(rvec)
                print(f"Rotation matrix:\n{R}")
                
                # 3x4 프로젝션 행렬 구성
                P = np.zeros((3, 4))
                P[:3, :3] = R
                P[:3, 3] = tvec.flatten()
                
                # 카메라 내부 파라미터 적용
                P = camera_matrix @ P
                print(f"RANSAC projection matrix:\n{P}")
                
                # 검증 (inliers만 사용)
                print(f"\n=== RANSAC Verification (Inliers Only) ===")
                total_error = 0
                inlier_count = 0
                for i, (point_2d, point_3d) in enumerate(zip(points_2d, points_3d)):
                    if i in inliers:
                        projected = self.project_3d_to_2d(point_3d, P)
                        if projected is not None:
                            error = np.sqrt((point_2d[0] - projected[0])**2 + (point_2d[1] - projected[1])**2)
                            total_error += error
                            inlier_count += 1
                            print(f"Inlier {inlier_count}: Original 2D({point_2d[0]:.1f}, {point_2d[1]:.1f}) -> Projected({projected[0]:.1f}, {projected[1]:.1f}) -> Error: {error:.2f}")
                        else:
                            print(f"Inlier {inlier_count}: Projection failed")
                
                if inlier_count > 0:
                    avg_error = total_error / inlier_count
                    print(f"Average projection error (inliers): {avg_error:.2f}")
                    
                    if avg_error < 20:  # 오차가 20픽셀 미만이면 RANSAC 결과 사용
                        print("Using RANSAC result (low error)")
                        return P
                    else:
                        print("RANSAC error too high, trying iterative solvePnP...")
                else:
                    print("No inliers found, trying iterative solvePnP...")
            else:
                print("RANSAC solvePnP failed or insufficient inliers, trying iterative solvePnP...")
                
        except Exception as e:
            print(f"RANSAC solvePnP error: {e}, trying iterative solvePnP...")
        
        # 일반적인 solvePnP 시도
        try:
            success, rvec, tvec = cv2.solvePnP(points_3d_cv, points_2d_cv, camera_matrix, dist_coeffs, 
                                              flags=cv2.SOLVEPNP_ITERATIVE)
            
            if success:
                print(f"Iterative solvePnP successful")
                print(f"Rotation vector: {rvec.flatten()}")
                print(f"Translation vector: {tvec.flatten()}")
                
                # 회전 벡터를 회전 행렬로 변환
                R, _ = cv2.Rodrigues(rvec)
                print(f"Rotation matrix:\n{R}")
                
                # 3x4 프로젝션 행렬 구성
                P = np.zeros((3, 4))
                P[:3, :3] = R
                P[:3, 3] = tvec.flatten()
                
                # 카메라 내부 파라미터 적용
                P = camera_matrix @ P
                print(f"Iterative projection matrix:\n{P}")
                
                # 검증
                print(f"\n=== Iterative solvePnP Verification ===")
                total_error = 0
                for i, (point_2d, point_3d) in enumerate(zip(points_2d, points_3d)):
                    projected = self.project_3d_to_2d(point_3d, P)
                    if projected is not None:
                        error = np.sqrt((point_2d[0] - projected[0])**2 + (point_2d[1] - projected[1])**2)
                        total_error += error
                        print(f"Point {i+1}: Original 2D({point_2d[0]:.1f}, {point_2d[1]:.1f}) -> Projected({projected[0]:.1f}, {projected[1]:.1f}) -> Error: {error:.2f}")
                    else:
                        print(f"Point {i+1}: Projection failed")
                
                avg_error = total_error / len(points_2d)
                print(f"Average projection error: {avg_error:.2f}")
                
                if avg_error < 30:  # 오차가 30픽셀 미만이면 iterative 결과 사용
                    print("Using iterative solvePnP result (low error)")
                    return P
                else:
                    print("Iterative solvePnP error too high, trying DLT...")
            else:
                print("Iterative solvePnP failed, trying DLT...")
                
        except Exception as e:
            print(f"Iterative solvePnP error: {e}, trying DLT...")
        
        # 기존 DLT 방법 (마지막 수단)
        print("Falling back to DLT method...")
        X = np.array([[X, Y, Z, 1] for (X, Y, Z) in points_3d]).T  # (4, N)
        u = np.array([[x, y, 1] for (x, y) in points_2d]).T        # (3, N)
        
        P = self.dlt_estimate(X, u)
        print(f"Final DLT projection matrix:\n{P}")
        
        # 검증: 입력 점들로 역투영 테스트
        print(f"\n=== DLT Verification ===")
        total_error = 0
        for i, (point_2d, point_3d) in enumerate(zip(points_2d, points_3d)):
            projected = self.project_3d_to_2d(point_3d, P)
            if projected is not None:
                error = np.sqrt((point_2d[0] - projected[0])**2 + (point_2d[1] - projected[1])**2)
                total_error += error
                print(f"Point {i+1}: Original 2D({point_2d[0]:.1f}, {point_2d[1]:.1f}) -> Projected({projected[0]:.1f}, {projected[1]:.1f}) -> Error: {error:.2f}")
            else:
                print(f"Point {i+1}: Projection failed")
        
        avg_error = total_error / len(points_2d)
        print(f"Average projection error: {avg_error:.2f}")
        
        return P

    def project_3d_to_2d(self, point_3d, P):
        """DLT 프로젝션 행렬을 사용한 3D→2D 투영"""
        X = np.array([point_3d[0], point_3d[1], point_3d[2], 1])
        print(f"Projecting 3D point: {point_3d} -> homogeneous: {X}")
        
        uvw = P @ X
        print(f"Projection result uvw: {uvw}")
        
        if abs(uvw[2]) > 1e-10:
            result = (uvw[0] / uvw[2], uvw[1] / uvw[2])
            print(f"Final 2D projection: {result}")
            return result
        else:
            print(f"project_3d_to_2d: uvw[2] too close to zero: {uvw[2]}")
            return None
    
    def update_dlt_transform_from_current_points(self):
        """현재 스텝의 점들과 모든 DLT 점들을 이용하여 DLT 변환 행렬 업데이트"""
        if self.dlt_transform_matrix is None:
            print("No existing DLT transform matrix to update")
            return False
        
        try:
            # 현재 스텝의 모든 점들 로드
            current_points = self.load_point_labels_from_h5(self.episode_idx, self.current_step)
            
            # base 이미지의 점들만 필터링
            base_points = [point for point in current_points if point['image_type'] == 'base']
            
            # 모든 DLT 점들과 현재 스텝의 점들을 합침
            all_points_2d = []
            all_points_3d = []
            
            # 먼저 모든 DLT 점들 추가
            if len(self.all_dlt_points_2d) > 0:
                all_points_2d.extend(self.all_dlt_points_2d)
                all_points_3d.extend(self.all_dlt_points_3d)
                print(f"Added {len(self.all_dlt_points_2d)} existing DLT points")
            
            # 현재 스텝의 점들 추가 (중복 제거)
            current_points_added = 0
            for point in base_points:
                # 현재 점이 기존 DLT 점들과 중복되지 않는지 확인
                is_duplicate = False
                for existing_2d in all_points_2d:
                    distance = ((point['x'] - existing_2d[0]) ** 2 + (point['y'] - existing_2d[1]) ** 2) ** 0.5
                    if distance < 5:  # 5픽셀 이내면 중복으로 간주
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    all_points_2d.append([point['x'], point['y']])
                    
                    # 3D 좌표 (현재 스텝의 로봇 EE pose 사용)
                    ee_pose = self.get_robot_ee_pose(self.episode_idx, self.current_step)
                    if ee_pose is not None:
                        ee_position = ee_pose[:3]  # x, y, z
                        all_points_3d.append(ee_position)
                        current_points_added += 1
                    else:
                        print(f"Could not get robot EE pose for point at ({point['x']:.1f}, {point['y']:.1f})")
                        # 2D 점도 제거
                        all_points_2d.pop()
            
            total_points = len(all_points_2d)
            if total_points < 6:
                print(f"Not enough total points ({total_points}) to update DLT transform (need at least 6)")
                print(f"Current step base points: {len(base_points)}")
                print(f"Existing DLT points: {len(self.all_dlt_points_2d)}")
                print(f"New points added: {current_points_added}")
                return False
            
            print(f"\n=== Updating DLT Transform from All Points ===")
            print(f"Total points available: {total_points}")
            print(f"  - Existing DLT points: {len(self.all_dlt_points_2d)}")
            print(f"  - Current step base points: {len(base_points)}")
            print(f"  - New points added: {current_points_added}")
            
            # DLT 변환 행렬 재계산
            new_transform = self.compute_dlt_transform(all_points_2d, all_points_3d)
            
            if new_transform is not None:
                self.dlt_transform_matrix = new_transform
                print("DLT transform matrix updated successfully!")
                return True
            else:
                print("Failed to update DLT transform matrix")
                return False
                
        except Exception as e:
            print(f"Error updating DLT transform: {e}")
            return False
    
    def auto_label_with_dlt(self, episode_idx: int, step_idx: int):
        """개선된 DLT 변환 행렬을 사용하여 자동 라벨링 (base 이미지만)"""
        if self.dlt_transform_matrix is None:
            print("No DLT transform matrix available")
            return []
        
        try:
            # 현재 스텝의 로봇 EE pose 가져오기
            ee_pose = self.get_robot_ee_pose(episode_idx, step_idx)
            if ee_pose is None:
                print("Could not get robot EE pose")
                return []
            
            # EE position (처음 3개 값)
            ee_position = ee_pose[:3]
            print(f"Robot EE position: {ee_position}")
            
            # 3D 점을 2D로 투영
            projected_2d = self.project_3d_to_2d(ee_position, self.dlt_transform_matrix)
            if projected_2d is None:
                print("Could not project EE position to 2D")
                return []
            
            x, y = projected_2d
            print(f"Projected 2D coordinates: ({x:.2f}, {y:.2f})")
            
            # 시간적 일관성을 위한 필터링
            filtered_x, filtered_y = self.apply_temporal_filter(episode_idx, step_idx, x, y)
            print(f"Filtered 2D coordinates: ({filtered_x:.2f}, {filtered_y:.2f})")
            
            # base 이미지 범위 확인
            data = self.get_step_data(episode_idx, step_idx)
            base_image = data['observation']['image']
            base_height, base_width = base_image.shape[:2]
            
            print(f"Base image dimensions: {base_width}x{base_height}")
            
            auto_points = []
            
            # base 이미지에만 투영 (wrist 이미지는 제외)
            if 0 <= filtered_x < base_width and 0 <= filtered_y < base_height:
                auto_points.append({
                    'x': float(filtered_x),
                    'y': float(filtered_y),
                    'image_type': 'base',
                    'step': step_idx,
                    'auto_labeled': True
                })
                print(f"Added auto point to base image: ({filtered_x:.2f}, {filtered_y:.2f})")
            else:
                print(f"Filtered point ({filtered_x:.2f}, {filtered_y:.2f}) outside base image bounds [{0}, {base_width}) x [{0}, {base_height})")
            
            print(f"Generated {len(auto_points)} auto-labeled points")
            return auto_points
            
        except Exception as e:
            print(f"Error in auto labeling with DLT: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def apply_temporal_filter(self, episode_idx: int, step_idx: int, x: float, y: float):
        """시간적 일관성을 위한 필터링 (이전 스텝들의 점들을 고려)"""
        try:
            # 이전 스텝들의 자동 라벨링된 점들을 확인
            previous_points = []
            window_size = 5  # 이전 5개 스텝 확인
            
            for prev_step in range(max(0, step_idx - window_size), step_idx):
                try:
                    prev_points = self.load_point_labels_from_h5(episode_idx, prev_step)
                    auto_points = [p for p in prev_points if p.get('auto_labeled', False) and p['image_type'] == 'base']
                    if auto_points:
                        # 가장 최근 자동 라벨링된 점 사용
                        latest_point = auto_points[-1]
                        previous_points.append((latest_point['x'], latest_point['y']))
                except:
                    continue
            
            if len(previous_points) >= 2:
                # 이전 점들의 평균 위치 계산
                avg_x = sum(p[0] for p in previous_points) / len(previous_points)
                avg_y = sum(p[1] for p in previous_points) / len(previous_points)
                
                # 현재 투영된 점과 이전 평균 점 사이의 거리 계산
                distance = np.sqrt((x - avg_x)**2 + (y - avg_y)**2)
                
                # 거리가 너무 크면 (이상치) 이전 평균 사용
                max_allowed_distance = 50.0  # 50픽셀
                if distance > max_allowed_distance:
                    print(f"Temporal filter: Large jump detected ({distance:.1f}px), using previous average")
                    return avg_x, avg_y
                else:
                    # 가중 평균 사용 (현재 점에 더 높은 가중치)
                    weight_current = 0.7
                    weight_previous = 0.3
                    filtered_x = weight_current * x + weight_previous * avg_x
                    filtered_y = weight_current * y + weight_previous * avg_y
                    print(f"Temporal filter: Applied weighted average (distance: {distance:.1f}px)")
                    return filtered_x, filtered_y
            else:
                # 이전 점이 충분하지 않으면 현재 점 그대로 사용
                print("Temporal filter: Not enough previous points, using current projection")
                return x, y
                
        except Exception as e:
            print(f"Temporal filter error: {e}, using current projection")
            return x, y
    
    def save_episode_with_labels(self):
        """현재 에피소드를 라벨링된 점들과 함께 새로운 h5py 파일로 저장"""
        try:
            episode_info = self.get_episode_info(self.episode_idx)
            num_steps = episode_info['num_steps']
            
            # 출력 파일명 생성
            output_filename = f"{self.save_path}/episode_{self.episode_idx:06d}_with_labels.h5"
            
            print(f"\n=== Saving Episode {self.episode_idx} with Labels ===")
            print(f"Output file: {output_filename}")
            print(f"Number of steps: {num_steps}")
            
            with h5py.File(output_filename, 'w') as output_f:
                # 메타데이터 복사
                for key, value in self.f.attrs.items():
                    output_f.attrs[key] = value
                
                # 추가 메타데이터
                output_f.attrs['labeled_episode'] = True
                output_f.attrs['original_episode_idx'] = self.episode_idx
                output_f.attrs['saved_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
                
                # 에피소드 그룹 생성
                episodes_group = output_f.create_group('episodes')
                episode_group = episodes_group.create_group(f'episode_{self.episode_idx:06d}')
                
                # 에피소드 메타데이터 복사
                episode_name = self.episode_names[self.episode_idx]
                original_episode_group = self.episodes_group[episode_name]
                for key, value in original_episode_group.attrs.items():
                    episode_group.attrs[key] = value
                
                # 각 스텝 복사 및 라벨링 정보 포함
                labeled_steps = 0
                total_points = 0
                
                for step_idx in range(num_steps):
                    step_name = f'step_{step_idx:06d}'
                    original_step_group = original_episode_group[step_name]
                    
                    # 스텝 그룹 생성
                    step_group = episode_group.create_group(step_name)
                    
                    # 모든 데이터 복사
                    for key in original_step_group.keys():
                        if key == 'observation':
                            # observation 그룹 특별 처리
                            obs_group = step_group.create_group('observation')
                            original_obs_group = original_step_group['observation']
                            
                            for obs_key in original_obs_group.keys():
                                # point_label, point_image는 나중에 처리하므로 건너뛰기
                                if obs_key in ['point_label', 'point_image', 'point_image_wrist']:
                                    continue
                                    
                                original_dataset = original_obs_group[obs_key]
                                if original_dataset.shape == ():  # 스칼라 데이터셋
                                    obs_group.create_dataset(obs_key, data=original_dataset[()])
                                else:  # 배열 데이터셋
                                    obs_group.create_dataset(obs_key, data=original_dataset[:])
                            
                            # 라벨링된 점들 추가 (기존 point_label 덮어쓰기)
                            labeled_points = self.load_point_labels_from_h5(self.episode_idx, step_idx)
                            if labeled_points:
                                # 점들을 numpy 배열로 변환
                                point_array = []
                                for point in labeled_points:
                                    image_type_code = 0 if point['image_type'] == 'base' else 1
                                    point_array.append([point['x'], point['y'], image_type_code])
                                
                                point_array = np.array(point_array, dtype=np.float32)
                                obs_group.create_dataset('point_label', data=point_array)
                                
                                # point_image 마스크 생성
                                base_image = obs_group['image'][:]
                                wrist_image = obs_group['wrist_image'][:]
                                base_height, base_width = base_image.shape[:2]
                                wrist_height, wrist_width = wrist_image.shape[:2]
                                
                                # base 이미지용 마스크
                                base_point_image = np.zeros((base_height, base_width), dtype=np.uint8)
                                # wrist 이미지용 마스크
                                wrist_point_image = np.zeros((wrist_height, wrist_width), dtype=np.uint8)
                                
                                # 라벨링된 점들을 마스크에 표시
                                for point in labeled_points:
                                    x, y = int(point['x']), int(point['y'])
                                    if point['image_type'] == 'base':
                                        if 0 <= x < base_width and 0 <= y < base_height:
                                            base_point_image[y, x] = 255
                                    elif point['image_type'] == 'wrist':
                                        if 0 <= x < wrist_width and 0 <= y < wrist_height:
                                            wrist_point_image[y, x] = 255
                                
                                obs_group.create_dataset('point_image', data=base_point_image)
                                obs_group.create_dataset('point_image_wrist', data=wrist_point_image)
                                
                                labeled_steps += 1
                                total_points += len(labeled_points)
                            else:
                                # 라벨링된 점이 없으면 빈 배열로 point_label 생성
                                empty_array = np.array([], dtype=np.float32).reshape(0, 3)
                                obs_group.create_dataset('point_label', data=empty_array)
                                
                                # 빈 point_image 마스크 생성
                                base_image = obs_group['image'][:]
                                wrist_image = obs_group['wrist_image'][:]
                                base_height, base_width = base_image.shape[:2]
                                wrist_height, wrist_width = wrist_image.shape[:2]
                                
                                empty_base_mask = np.zeros((base_height, base_width), dtype=np.uint8)
                                empty_wrist_mask = np.zeros((wrist_height, wrist_width), dtype=np.uint8)
                                
                                obs_group.create_dataset('point_image', data=empty_base_mask)
                                obs_group.create_dataset('point_image_wrist', data=empty_wrist_mask)
                        else:
                            # 다른 데이터는 그대로 복사
                            original_dataset = original_step_group[key]
                            if original_dataset.shape == ():  # 스칼라 데이터셋
                                step_group.create_dataset(key, data=original_dataset[()])
                            else:  # 배열 데이터셋
                                step_group.create_dataset(key, data=original_dataset[:])
                
                print(f"Successfully saved episode {self.episode_idx} to {output_filename}")
                print(f"Labeled steps: {labeled_steps}/{num_steps}")
                print(f"Total labeled points: {total_points}")
                
                if labeled_steps > 0:
                    print(f"Labeled steps: {labeled_steps}/{num_steps} ({labeled_steps/num_steps*100:.1f}%)")
                    print(f"Average points per labeled step: {total_points/labeled_steps:.1f}")
                else:
                    print("No labeled points found in this episode")
                
        except Exception as e:
            print(f"Error saving episode with labels: {e}")
            import traceback
            traceback.print_exc()
    
    def close(self):
        """파일 핸들러 닫기"""
        self.f.close()

def main(
    file_path: str,
    save_path: str,
    episode_idx: int = 0,
    step_idx: Optional[int] = None,
    start_step: int = 0,
    end_step: Optional[int] = None,
    interactive: bool = True,
    save_images: bool = False
):
    """
    h5py 파일을 시각화
    
    Args:
        file_path: h5py 파일 경로
        save_path: 저장 경로
        episode_idx: 시각화할 에피소드 인덱스
        step_idx: 특정 스텝만 시각화 (None이면 인터랙티브 모드)
        start_step: 인터랙티브 모드 시작 스텝 (0부터 시작, -1이면 마지막 스텝)
        end_step: 인터랙티브 모드 종료 스텝
        interactive: 인터랙티브 모드 사용 여부
        save_images: 이미지 저장 여부
    """
    # Interactive mode 활성화
    plt.ion()
    
    print(f"Loading h5py dataset: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return
    
    visualizer = H5pyVisualizer(file_path, save_path)
    
    try:
        # 파일 정보 출력
        print(f"Dataset metadata:")
        for key, value in visualizer.f.attrs.items():
            print(f"  {key}: {value}")
        
        print(f"\nNumber of episodes: {len(visualizer.episode_names)}")
        
        # 특정 스텝만 시각화
        if step_idx is not None:
            print(f"\nVisualizing Episode {episode_idx}, Step {step_idx}")
            save_path = f"visualization_output/ep{episode_idx}_step{step_idx}.png" if save_images else None
            if save_images:
                os.makedirs("visualization_output", exist_ok=True)
            visualizer.visualize_step(episode_idx, step_idx, save_path)
        
        # 인터랙티브 모드
        elif interactive:
            # start_step이 -1이면 가장 마지막 스텝으로 설정
            if start_step == -1:
                episode_info = visualizer.get_episode_info(episode_idx)
                num_steps = episode_info['num_steps']
                start_step = num_steps - 1
                print(f"start_step is -1, setting to last step: {start_step}")
            
            visualizer.interactive_visualizer(episode_idx, start_step, end_step)
        
        # 배치 시각화 (start_step부터 end_step까지)
        else:
            # start_step이 -1이면 가장 마지막 스텝으로 설정
            if start_step == -1:
                episode_info = visualizer.get_episode_info(episode_idx)
                num_steps = episode_info['num_steps']
                start_step = num_steps - 1
                print(f"start_step is -1, setting to last step: {start_step}")
            
            if end_step is None:
                episode_info = visualizer.get_episode_info(episode_idx)
                end_step = episode_info['num_steps'] - 1
            
            print(f"\nBatch visualization: Episode {episode_idx}, Steps {start_step} to {end_step}")
            
            if save_images:
                os.makedirs("visualization_output", exist_ok=True)
            
            for step in range(start_step, end_step + 1):
                try:
                    save_path = f"visualization_output/ep{episode_idx}_step{step}.png" if save_images else None
                    visualizer.visualize_step(episode_idx, step, save_path)
                    print(f"Processed step {step}")
                except Exception as e:
                    print(f"Error at step {step}: {e}")
                    continue
    
    finally:
        visualizer.close()

if __name__ == "__main__":
    tyro.cli(main) 