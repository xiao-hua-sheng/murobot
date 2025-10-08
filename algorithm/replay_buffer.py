import numpy as np
from collections import deque
import random
from typing import List, Tuple, Optional


class ExperienceReplayBuffer:
    def __init__(self, max_size=10000, batch_size=64):
        """
        自定义经验回放缓冲区

        参数:
            max_size: 缓冲区最大容量
            batch_size: 采样批量大小
        """
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_size)

        # 存储轨迹数据的结构
        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.returns = []
        self.advantages = []

        # 当前缓冲区大小
        self.current_size = 0

    def add(self, state, action, old_log_prob, return_, advantage):
        """
        添加单条经验到缓冲区

        参数:
            state: 状态
            action: 动作
            old_log_prob: 旧策略的对数概率
            return_: 回报
            advantage: 优势函数值
        """
        # 如果缓冲区已满，移除最旧的经验
        if self.current_size >= self.max_size:
            self.states.pop(0)
            self.actions.pop(0)
            self.old_log_probs.pop(0)
            self.returns.pop(0)
            self.advantages.pop(0)
        else:
            self.current_size += 1

        # 添加新经验
        self.states.append(state)
        self.actions.append(action)
        self.old_log_probs.append(old_log_prob)
        self.returns.append(return_)
        self.advantages.append(advantage)

    def add_batch(self, states, actions, old_log_probs, returns, advantages):
        """
        批量添加经验到缓冲区

        参数:
            states: 状态批次
            actions: 动作批次
            old_log_probs: 旧策略的对数概率批次
            returns: 回报批次
            advantages: 优势函数值批次
        """
        batch_size = len(states)
        for i in range(batch_size):
            self.add(
                states[i],
                actions[i],
                old_log_probs[i],
                returns[i],
                advantages[i]
            )

    def sample(self, batch_size=None):
        """
        从缓冲区随机采样一个批次

        参数:
            batch_size: 批次大小，如果为None则使用默认批次大小

        返回:
            包含状态、动作、旧对数概率、回报和优势的元组
        """
        if batch_size is None:
            batch_size = self.batch_size

        # 确保采样大小不超过缓冲区大小
        batch_size = min(batch_size, self.current_size)

        # 随机采样索引
        indices = random.sample(range(self.current_size), batch_size)

        # 收集采样数据
        batch_states = [self.states[i] for i in indices]
        batch_actions = [self.actions[i] for i in indices]
        batch_old_log_probs = [self.old_log_probs[i] for i in indices]
        batch_returns = [self.returns[i] for i in indices]
        batch_advantages = [self.advantages[i] for i in indices]

        # batch_states = torch.stack(batch_states)
        # batch_actions = torch.stack(batch_actions)
        # batch_old_log_probs = torch.stack(batch_old_log_probs)
        # batch_returns = torch.stack(batch_returns)
        # batch_advantages = torch.stack(batch_advantages)

        return batch_states, batch_actions, batch_old_log_probs, batch_returns, batch_advantages

    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.returns = []
        self.advantages = []
        self.current_size = 0

    def __len__(self):
        """返回缓冲区当前大小"""
        return self.current_size

    def is_full(self):
        """检查缓冲区是否已满"""
        return self.current_size >= self.max_size

    def get_memory_usage(self):
        """估算内存使用量（MB）"""
        if self.current_size == 0:
            return 0

        # 估算单个样本的内存使用
        sample_memory = (
                self.states[0].element_size() * self.states[0].nelement() +
                self.actions[0].element_size() * self.actions[0].nelement() +
                self.old_log_probs[0].element_size() * self.old_log_probs[0].nelement() +
                self.returns[0].element_size() * self.returns[0].nelement() +
                self.advantages[0].element_size() * self.advantages[0].nelement()
        )

        # 转换为MB
        total_memory = sample_memory * self.current_size / (1024 * 1024)
        return total_memory


class ReplayBuffer:
    def __init__(self, capacity: int = 1000000):
        """
        初始化经验回放缓冲区

        Args:
            capacity: 缓冲区的最大容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.size = 0

    def add(self, state: np.ndarray, action: np.ndarray,
            reward: float, next_state: np.ndarray, done: bool):
        """
        添加一条经验到缓冲区

        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.size = len(self.buffer)

    def sample(self, batch_size: int) -> Tuple:
        """
        从缓冲区中随机采样一批经验

        Args:
            batch_size: 批量大小

        Returns:
            批量的状态、动作、奖励、下一个状态和完成标志
        """
        if self.size < batch_size:
            batch_size = self.size

        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.size = 0

    def __len__(self):
        """返回缓冲区当前大小"""
        return self.size

    def size(self):
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """带优先级的经验回放缓冲区"""

    def __init__(self, capacity: int = 1000000, alpha: float = 0.6, beta: float = 0.4):
        """
        初始化优先级经验回放缓冲区

        Args:
            capacity: 缓冲区的最大容量
            alpha: 优先级指数 (0表示均匀采样)
            beta: 重要性采样权重调整参数
        """
        super().__init__(capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

    def add(self, state: np.ndarray, action: np.ndarray,
            reward: float, next_state: np.ndarray, done: bool):
        """
        添加一条经验到缓冲区，初始优先级设为当前最大优先级

        Args:
            state: 当前状态
            action: 采取的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
        """
        super().add(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)

    def sample(self, batch_size: int) -> Tuple:
        """
        根据优先级采样一批经验

        Args:
            batch_size: 批量大小

        Returns:
            批量的状态、动作、奖励、下一个状态、完成标志和重要性权重
        """
        if self.size < batch_size:
            batch_size = self.size

        # 计算采样概率
        priorities = np.array(self.priorities, dtype=np.float32) ** self.alpha
        probs = priorities / priorities.sum()

        # 根据概率采样索引
        indices = np.random.choice(self.size, batch_size, p=probs)

        # 获取对应的经验
        samples = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*samples)

        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            np.array(weights, dtype=np.float32)
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        更新经验的优先级

        Args:
            indices: 需要更新优先级的经验索引
            priorities: 新的优先级值
        """
        for idx, priority in zip(indices, priorities):
            # 添加一个小值避免优先级为0
            self.priorities[idx] = priority + 1e-6
            self.max_priority = max(self.max_priority, priority)


class DualExperienceReplayBuffer:
    def __init__(self, total_capacity: int, grasp_ratio: float = 0.5):
        """
        双经验池初始化

        Args:
            total_capacity: 总容量
            grasp_ratio: 成功夹取经验在混合批次中的比例
        """
        self.common_buffer = []  # 普通经验池
        self.grasp_buffer = []  # 成功夹取经验池
        self.total_capacity = total_capacity
        self.grasp_ratio = grasp_ratio

    def add(self, state, action, reward, next_state, done, is_grasp: bool):
        """
        添加经验到相应的缓冲区

        Args:
            is_grasp: 是否成功夹取
        """
        experience = (state, action, reward, next_state, done)

        if is_grasp:
            # 添加到成功经验池
            if len(self.grasp_buffer) >= self.total_capacity // 2:
                self.grasp_buffer.pop(0)
            self.grasp_buffer.append(experience)
        else:
            # 添加到普通经验池
            if len(self.common_buffer) >= self.total_capacity - len(self.grasp_buffer):
                self.common_buffer.pop(0)
            self.common_buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        """
        从两个缓冲区按比例采样并合并

        Args:
            batch_size: 批量大小

        Returns:
            批量的状态、动作、奖励、下一个状态和完成标志
        """
        # 计算每个缓冲区的采样数量
        grasp_batch_size = int(batch_size * self.grasp_ratio)
        common_batch_size = batch_size - grasp_batch_size

        # 调整采样数量以防某个缓冲区数据不足
        grasp_available = min(len(self.grasp_buffer), grasp_batch_size)
        common_available = min(len(self.common_buffer), common_batch_size)

        # 如果某个缓冲区数据不足，从另一个缓冲区补充
        if grasp_available < grasp_batch_size:
            common_available = min(len(self.common_buffer), common_batch_size + (grasp_batch_size - grasp_available))
        elif common_available < common_batch_size:
            grasp_available = min(len(self.grasp_buffer), grasp_batch_size + (common_batch_size - common_available))

        # 从两个缓冲区采样
        grasp_batch = random.sample(self.grasp_buffer, grasp_available) if grasp_available > 0 else []
        common_batch = random.sample(self.common_buffer, common_available) if common_available > 0 else []

        # 合并批次
        combined_batch = grasp_batch + common_batch

        # 如果总经验数不足，减少批次大小
        if len(combined_batch) < batch_size:
            print(f"Warning: Not enough experiences. Requested {batch_size}, got {len(combined_batch)}")

        # 解压批次
        states, actions, rewards, next_states, dones = zip(*combined_batch) if combined_batch else ([], [], [], [], [])

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    @property
    def size(self) -> int:
        """返回总经验数"""
        return len(self.common_buffer) + len(self.grasp_buffer)

    def size_info(self):
        return len(self.common_buffer), len(self.grasp_buffer)