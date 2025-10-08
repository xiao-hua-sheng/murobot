import torch
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F

from typing import List, Tuple
from torch.utils.tensorboard import SummaryWriter


class PPO:
    """PPO算法实现"""

    def __init__(self,
                 agent: torch.nn.Module,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 entropy_coef: float = 0.01,
                 lr: float = 3e-4,
                 epochs: int = 5):

        self.agent = agent
        self.optimizer = optim.Adam(agent.parameters(), lr=lr)

        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.writer = SummaryWriter(f'runs/unitreego2_loss_{int(time.time())}')

    def compute_returns_and_advantages(self,
                                       rewards: List[float],
                                       dones: List[bool],
                                       values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算GAE和returns"""
        returns = []
        advantages = []
        last_advantage = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage
            returns.insert(0, advantage + values[t])
            advantages.insert(0, advantage)
            next_value = values[t]
            last_advantage = advantage

        returns = torch.tensor(returns)
        advantages = torch.tensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self,
               states: List[np.ndarray],
               actions: List[np.ndarray],
               rewards: List,
               dones: List,
               old_log_probs: List[float],
               episode: int
               ):

        # 转换为Tensor
        states = torch.as_tensor(states, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.float32)
        old_log_probs = torch.as_tensor(old_log_probs, dtype=torch.float32).squeeze()
        # rewards = torch.as_tensor(rewards, dtype=torch.float32)
        # dones = torch.as_tensor(dones)

        advantages = []
        returns = []
        # 计算旧值
        with torch.no_grad():
            old_values = self.agent.get_value(states)

        # 计算GAE和returns
        for i in range(len(rewards)):
            returns_, advantages_ = self.compute_returns_and_advantages(
                rewards[i], dones[i], old_values[i].numpy()
            )
            returns.append(returns_)
            advantages.append(advantages_)

        returns = torch.stack(returns)
        advantages = torch.stack(advantages)
        # 计算新策略的概率和值
        log_probs, values, entropy = self.agent.evaluate(states, actions)

        # 计算比率
        ratios = (log_probs - old_log_probs).exp()

        # 策略损失
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值函数损失
        value_loss = F.mse_loss(values, returns)

        # 熵奖励
        entropy_loss = -entropy.mean() * self.entropy_coef

        # 总损失
        total_loss = policy_loss + value_loss + entropy_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
        self.optimizer.step()

        self.writer.add_scalar("policy_loss", policy_loss, episode)
        self.writer.add_scalar("entropy_loss", entropy_loss, episode)
        self.writer.add_scalar("value_loss", value_loss, episode)
