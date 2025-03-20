import os
import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class VideoRecordingCallback(BaseCallback):
    """
    自定义回调：在训练过程中录制视频
    每个 episode 结束时，将收集到的帧保存为视频文件，文件名中包含 episode 编号。
    """
    def __init__(self, video_folder, video_length=500, fps=30, verbose=0):
        super(VideoRecordingCallback, self).__init__(verbose)
        self.video_folder = video_folder
        os.makedirs(self.video_folder, exist_ok=True)
        self.video_length = video_length  # 每个视频录制的最大帧数
        self.fps = fps
        self.frames = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # 从环境中获取当前帧
        frame = self.training_env.env_method("render", render_mode="rgb_array")
        # 如果返回的是列表（多个环境），取第一个
        if isinstance(frame, list):
            frame = frame[0]
        self.frames.append(frame)
        return True

    def _on_rollout_end(self) -> None:
        # 当 rollout 结束时，不保存视频，等待 episode 结束
        pass

    def _on_episode_end(self) -> None:
        # 每个 episode 结束时，将收集的帧保存为视频文件
        if len(self.frames) > 0:
            height, width, channels = self.frames[0].shape
            filename = os.path.join(self.video_folder, f"episode_{self.episode_count}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(filename, fourcc, self.fps, (width, height))
            for frame in self.frames:
                # 将 RGB 转为 BGR 格式（OpenCV要求）
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame_bgr)
            video_writer.release()
            print(f"Episode {self.episode_count} 视频保存成功：{filename}")
            self.episode_count += 1
            self.frames = []  # 清空帧缓存

