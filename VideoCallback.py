import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback


class MetricsLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricsLoggingCallback, self).__init__(verbose)
        # 用于记录每一步的指标
        self.steps = []
        self.avg_delays = []
        self.hit_rates = []
        self.avg_rewards = []

    def _on_step(self) -> bool:
        # 从 locals 中获取 infos（infos 是一个列表，可能包含多个 info 字典）
        infos = self.locals.get("infos", [])
        # 如果 infos 非空，则提取你需要的指标
        for info in infos:
            if info is not None:
                # 确保 info 中包含指标字段
                if "avg_delay" in info:
                    self.avg_delays.append(info["avg_delay"])
                if "hit_rate" in info:
                    self.hit_rates.append(info["hit_rate"])
                if "avg_reward" in info:
                    self.avg_rewards.append(info["avg_reward"])
                # 记录当前步数（这里用 num_timesteps）
                self.steps.append(self.num_timesteps)
        return True

    def plot_metrics(self):
        plt.figure(figsize=(10, 5))

        plt.plot(self.steps, self.avg_delays, label="Avg Delay", color='blue')
        plt.plot(self.steps, self.hit_rates, label="Hit Rate", color='green')
        plt.plot(self.steps, self.avg_rewards, label="Avg Reward", color='red')

        plt.xlabel("Steps")
        plt.ylabel("Metric Value")
        plt.title("Training Metrics Over Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

