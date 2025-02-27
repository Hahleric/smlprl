
import numpy as np
import torch as t
# 假设已有的 Vehicle, RSU, Crossroad 类（参考前面代码）
# Vehicle 类示例（修正 generate_request 中 interest_vector 的引用）
class Vehicle:
    def __init__(self, user_id, position, speed, interest_vector):
        """
        :param user_id: 车辆或用户的唯一标识
        :param position: 车辆初始位置（二维坐标）
        :param speed: 车辆速度向量（二维）
        :param interest_vector: 兴趣向量（例如代表请求电影的概率分布，维度与候选集合长度一致）
        """
        self.user_id = user_id
        self.position = np.array(position, dtype=float)
        self.speed = np.array(speed, dtype=float)
        self.interest_vector = np.array(interest_vector, dtype=float)


    def update_position(self, dt=1):
        self.position += self.speed * dt

    def get_bandwidth(self, rsu_position, B0=10.0, alpha=0.1):
        distance = np.linalg.norm(self.position - rsu_position)
        bandwidth = B0 * np.exp(-alpha * distance) + np.random.normal(0, 0.5)
        return max(bandwidth, 0)
    def get_position(self):
        return self.position

# RSU 类（用于带宽计算，不含缓存状态，缓存由环境管理）
class RSU:
    def __init__(self, position, B0=10.0, alpha=0.1):
        self.position = np.array(position)
        self.B0 = B0
        self.alpha = alpha

    def compute_bandwidth(self, vehicle_position):
        vehicle_position = np.array(vehicle_position)
        distance = np.linalg.norm(vehicle_position - self.position)
        bandwidth = self.B0 * np.exp(-self.alpha * distance) + np.random.normal(0, 0.5)
        return max(bandwidth, 0.0)

# Crossroad 类（管理车辆生成、更新、删除等）
class Crossroad:
    def __init__(self, width, height, rsu_position=None, B0=10.0, alpha=0.1, spawn_rate=0.2, user_interest_dict=None):
        self.width = width
        self.height = height
        if rsu_position is None:
            rsu_position = [width / 2, height / 2]
        self.rsu = RSU(rsu_position, B0, alpha)
        self.vehicles = []
        self.spawn_rate = spawn_rate  # 每个时间步生成新车辆的期望数量
        self.vehicle_counter = 0
        # 保存预处理得到的用户兴趣向量字典
        self.user_interest_dict = user_interest_dict
        self.max_distance = np.linalg.norm([width, height])

    def generate_vehicle(self, user_id=None, position=None, speed=None, interest_vector=None):
        if user_id is None:
            # 如果提供了兴趣字典，则随机选择一个用户ID；否则使用内部计数器
            if self.user_interest_dict is not None:
                user_id = np.random.choice(list(self.user_interest_dict.keys()))
            else:
                user_id = self.vehicle_counter
                self.vehicle_counter += 1

        if position is None:
            # 随机生成在区域内部的位置
            position = np.random.uniform([0, 0], [self.width, self.height])
        if speed is None:
            speed = np.random.uniform(-1, 1, size=2)
        if interest_vector is None:
            # 如果已经提供了预处理的兴趣字典，则根据 user_id 获取 interest_vector
            if self.user_interest_dict is not None and user_id in self.user_interest_dict:
                interest_vector = self.user_interest_dict[user_id]
            else:
                # 否则随机生成（假设候选集合大小为10）
                interest_vector = np.random.rand(10)
                interest_vector /= interest_vector.sum()
        vehicle = Vehicle(user_id, position, speed, interest_vector)
        self.vehicles.append(vehicle)
        return vehicle

    def spawn_vehicle_from_edge(self):
        edge = np.random.choice(['left', 'right', 'top', 'bottom'])
        if edge == 'left':
            x = 0
            y = np.random.uniform(0, self.height)
            speed = [np.random.uniform(0.5, 1.5), np.random.uniform(-1, 1)]
        elif edge == 'right':
            x = self.width
            y = np.random.uniform(0, self.height)
            speed = [np.random.uniform(-1.5, -0.5), np.random.uniform(-1, 1)]
        elif edge == 'top':
            y = self.height
            x = np.random.uniform(0, self.width)
            speed = [np.random.uniform(-1, 1), np.random.uniform(-1.5, -0.5)]
        else:
            y = 0
            x = np.random.uniform(0, self.width)
            speed = [np.random.uniform(-1, 1), np.random.uniform(0.5, 1.5)]
        position = [x, y]
        return self.generate_vehicle(position=position, speed=speed)

    def update_vehicles(self, dt=1.0):
        for vehicle in self.vehicles:
            vehicle.update_position(dt)

    def remove_exited_vehicles(self):
        remaining = []
        for vehicle in self.vehicles:
            x, y = vehicle.position
            if 0 <= x <= self.width and 0 <= y <= self.height:
                remaining.append(vehicle)
        self.vehicles = remaining

    def spawn_new_vehicles(self):
        num_to_spawn = int(self.spawn_rate)
        for _ in range(num_to_spawn):
            self.spawn_vehicle_from_edge()
        remainder = self.spawn_rate - num_to_spawn
        if np.random.rand() < remainder:
            self.spawn_vehicle_from_edge()

    def simulate_step(self, dt=1.0):
        self.update_vehicles(dt)
        self.remove_exited_vehicles()
        self.spawn_new_vehicles()
        # 返回车辆的带宽信息（用于调试或状态构造）
        bandwidths = {vehicle.user_id: vehicle.get_bandwidth(self.rsu.position) for vehicle in self.vehicles}
        return bandwidths