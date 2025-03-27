import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class MetricsLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricsLoggingCallback, self).__init__(verbose)
        self.steps = []
        self.avg_delays = []
        self.hit_rates = []
        self.avg_rewards = []
        self.episode_counter = 0

        # 缓存当前 episode 的指标
        self.current_episode_delays = []
        self.current_episode_hits = []
        self.current_episode_rewards = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if info is not None:
                # 累积 episode 数据
                if "avg_delay" in info:
                    self.current_episode_delays.append(info["avg_delay"])
                if "hit_rate" in info:
                    self.current_episode_hits.append(info["hit_rate"])
                if "avg_reward" in info:
                    self.current_episode_rewards.append(info["avg_reward"])

            if done:  # 这个 episode 结束了
                # 计算 episode 平均
                self.episode_counter += 1
                self.steps.append(self.episode_counter)
                if self.current_episode_delays:
                    self.avg_delays.append(np.mean(self.current_episode_delays))
                    self.hit_rates.append(np.mean(self.current_episode_hits))
                    self.avg_rewards.append(np.mean(self.current_episode_rewards))

                # 清空缓存
                self.current_episode_delays.clear()
                self.current_episode_hits.clear()
                self.current_episode_rewards.clear()

        return True

    def plot_metrics(self):
        plt.figure(figsize=(10, 5))

        plt.plot(self.steps, self.avg_delays, label="Avg Delay", color='blue', marker='o', markersize=2, linewidth=1)
        plt.plot(self.steps, self.hit_rates, label="Hit Rate", color='green', marker='o', markersize=2, linewidth=1)
        plt.plot(self.steps, self.avg_rewards, label="Avg Reward", color='red', marker='o', markersize=2, linewidth=1)

        plt.xlabel("episodes")
        plt.ylabel("Metric Value")
        plt.title("Training Metrics per Episode")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

