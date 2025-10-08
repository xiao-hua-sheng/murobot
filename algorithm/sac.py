import torch
import torch.optim as optim
import numpy as np
import time
import torch.nn.functional as F

from typing import List, Tuple, Optional
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SAC:
    """Soft Actor-Critic (SAC) 算法实现"""

    def __init__(self,
                 agent: torch.nn.Module,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 lr: float = 3e-4,
                 reward_scale: float = 1.0,
                 target_entropy: Optional[float] = None):

        self.agent = agent

        # 优化器
        self.policy_optimizer = optim.Adam(self.agent.actor.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.agent.critic1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.agent.critic2.parameters(), lr=lr)

        # 自动调整温度参数alpha
        self.alpha = alpha
        if target_entropy is None:
            self.target_entropy = -torch.prod(torch.Tensor([self.agent.action_dim])).item()
        else:
            self.target_entropy = target_entropy

        self.log_alpha = torch.zeros(1,device=device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # 超参数
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale

    def update(self, batch):
        """使用一批经验数据更新网络"""
        states, actions, rewards, next_states, dones = batch

        # 转换为Tensor
        states = torch.as_tensor(states, dtype=torch.float32).to(device)
        actions = torch.as_tensor(actions, dtype=torch.float32).to(device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.as_tensor(next_states, dtype=torch.float32).to(device)
        dones = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        # 更新Q函数
        with torch.no_grad():
            next_actions, next_log_probs = self.agent.actor.sample(next_states)
            next_q1, next_q2 = self.agent.get_target_q_values(next_states, next_actions)

            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards * self.reward_scale + (1 - dones) * self.gamma * next_q

        # 计算当前Q值
        current_q1 = self.agent.critic1(states, actions)
        current_q2 = self.agent.critic2(states, actions)

        # Q函数损失
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        # 更新Q网络
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # 更新策略网络
        new_actions, log_probs = self.agent.actor.sample(states)
        q1_new = self.agent.critic1(states, new_actions)
        q2_new = self.agent.critic2(states, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        policy_loss = (self.alpha * log_probs - min_q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # 更新温度参数alpha
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # 软更新目标网络
        self.soft_update(self.agent.critic1, self.agent.target_critic1)
        self.soft_update(self.agent.critic2, self.agent.target_critic2)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha
        }

    def soft_update(self, local_model, target_model):
        """软更新目标网络参数"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)