import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Tuple
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    """策略网络（Actor）"""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: list):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], output_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std


class ValueNetwork(nn.Module):
    """价值网络（Critic）"""

    def __init__(self, input_dim: int, hidden_dim: list):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.ReLU(),
            nn.Linear(hidden_dim[2], 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PPOAgent(nn.Module):
    """PPO智能体"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: dict):
        super().__init__()
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim["policy_hdim"])
        self.value = ValueNetwork(state_dim, hidden_dim["v_hdim"])

    def get_action(self, state: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        state_tensor = torch.FloatTensor(state)
        mean, std = self.policy(state_tensor)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.detach(), log_prob.detach()

    def get_action_(self, state: np.ndarray):
        state_tensor = torch.FloatTensor(state)
        mean, std = self.policy(state_tensor)
        action = mean.detach()
        return action

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.policy(states)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        values = self.value(states)
        return log_probs, values, entropy

    def get_value(self, states: torch.Tensor) -> torch.Tensor:
        return self.value(states)


class GaussianPolicy(nn.Module):
    """高斯策略网络，输出均值和标准差"""

    def __init__(self, state_dim, action_dim, hidden_dim: list, log_std_min=-20, log_std_max=2):
        super(GaussianPolicy, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])

        self.mean_layer = nn.Linear(hidden_dim[2], action_dim)
        self.log_std_layer = nn.Linear(hidden_dim[2], action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        """从策略中采样动作并计算log概率"""
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # 重参数化技巧
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 重参数化采样
        action = torch.tanh(x_t)

        # 计算log概率
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    """Q函数网络"""

    def __init__(self, state_dim, action_dim, hidden_dim: list):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.fc4 = nn.Linear(hidden_dim[2], 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class VNetwork(nn.Module):
    """Q函数网络"""

    def __init__(self, state_dim, hidden_dim: list):
        super(VNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.fc4 = nn.Linear(hidden_dim[2], 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class SACAgent(nn.Module):
    """SAC算法的Actor-Critic网络"""

    def __init__(self, state_dim, action_dim, hidden_dim: list):
        super(SACAgent, self).__init__()
        self.action_dim = action_dim

        # Actor网络
        self.actor = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)

        # 两个Critic网络（Q函数）
        self.critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)

        # 一个价值网络（V函数）
        # self.value_net = VNetwork(state_dim, hidden_dim).to(device)

        # 目标网络
        self.target_critic1 = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic2 = QNetwork(state_dim, action_dim, hidden_dim).to(device)

        # 初始化目标网络参数与主网络相同
        self.hard_update(self.target_critic1, self.critic1)
        self.hard_update(self.target_critic2, self.critic2)

    def hard_update(self, target, source):
        """硬更新目标网络参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def get_action(self, state, deterministic=False):
        """获取动作（用于推理）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.actor(state)

        if deterministic:
            return torch.tanh(mean).detach().cpu().numpy()[0]
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            return action.detach().cpu().numpy()[0]

    def get_q_values(self, state, action):
        """获取两个Q网络的值"""
        return self.critic1(state, action), self.critic2(state, action)

    def get_target_q_values(self, state, action):
        """获取两个目标Q网络的值"""
        return self.target_critic1(state, action), self.target_critic2(state, action)

    def sample(self, state):
        """采样动作并计算log概率（用于训练）"""
        return self.actor.sample(state)