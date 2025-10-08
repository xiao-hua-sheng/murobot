import os.path
import torch
import time
import numpy as np

from typing import List, Tuple
from agent.agent import PPOAgent, SACAgent
from algorithm.ppo import PPO
from rewards.reward_go2 import reward_go2
from algorithm.replay_buffer import ExperienceReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tools.config import ConfigLoader
from envs.unitreego2 import UnitreeGo2Env


class PPOTrainer:
    def __init__(self, configs: dict):
        # 初始化环境
        self.render = configs["training"]["render"]
        self.env_name = configs["environment"]["name"]
        self.robot_path = configs["environment"]["robot_path"]
        self.pre_model_path = configs["training"]["pre_model_path"]
        self.save_path = configs["training"]["output_dir"]

        # 加载环境
        self.env = UnitreeGo2Env(self.robot_path)
        self.state_dim = self.env.get_observation_dim()
        self.action_dim = self.env.get_action_dim()
        self.agent_network = configs["agent_network"]

        # 初始化Agent
        self.agent = PPOAgent(self.state_dim, self.action_dim, hidden_dim=self.agent_network)

        if self.pre_model_path:
            checkpoint = torch.load(self.pre_model_path)
            self.agent.load_state_dict(checkpoint)

        # 初始化rl算法
        self.ppo = PPO(
            agent=self.agent,
            gamma=configs["algorithm_ppo"].get('gamma', 0.99),
            clip_epsilon=configs["algorithm_ppo"].get('clip_epsilon', 0.2),
            entropy_coef=configs["algorithm_ppo"].get('entropy_coef', 0.01),
            lr=configs["algorithm_ppo"].get('learning_rate', 1e-4)
        )

        # 训练参数
        self.max_episodes = configs["training"].get('max_episodes', 100000)
        self.max_steps = configs["training"].get('max_steps', 200)
        self.batch_size = configs['training']['batch_size']
        self.buffer_size = configs['replay_buffer']['buffer_size']

        # 初始化经验池
        self.replay_buff = ExperienceReplayBuffer(max_size=self.buffer_size, batch_size=self.batch_size)

        self.writer = SummaryWriter(f'runs/{self.env_name}_reward_{int(time.time())}')


    def collect_trajectory(self) -> Tuple[List, List, List, List, List, List]:
        """收集单条轨迹数据"""
        states, actions, rewards, dones, old_log_probs, reward_infos = [], [], [], [], [], []
        state = self.env.reset()

        for _ in range(self.max_steps):
            with torch.no_grad():
                action, old_log_prob = self.agent.get_action(state)

            next_state, raw_reward, done, info, reward_info = self.env.step(action.numpy())

            # 存储转换数据
            states.append(state)
            actions.append(action.numpy())
            rewards.append(raw_reward)
            dones.append(done)
            old_log_probs.append(old_log_prob.numpy().flatten())
            reward_infos.append(reward_info)

            state = next_state
            # if done or truncated:
            #     break

            if self.render:
                self.env.render()
        return states, actions, rewards, dones, old_log_probs, reward_infos

    def train(self):
        """主训练循环"""
        for episode in range(self.max_episodes):
            # 数据收集
            states, actions, rewards, dones, old_log_probs, reward_infos = self.collect_trajectory()
            # 奖励日志记录
            episode_reward = sum(rewards) / len(rewards)
            reward_infos = sum(np.array(reward_infos)) / len(rewards)
            print(f"Episode {episode + 1}/{self.max_episodes}, Total Reward: {episode_reward:.2f}, "
                  f"mv_Reward: {reward_infos[0]:.3f}, h_Reward: {reward_infos[1]:.3f}, ac_Reward: {reward_infos[2]:.3f}")
            self.writer.add_scalar('Episode_Reward', episode_reward, episode)
            self.writer.add_scalar('movement_reward', reward_infos[0], episode)
            self.writer.add_scalar('height_reward', reward_infos[1], episode)
            self.writer.add_scalar('action_constraint_reward', reward_infos[2], episode)
            # 保存轨迹并从经验池中采样
            self.replay_buff.add(states, actions, rewards, dones, old_log_probs)

            states, actions, rewards, dones, old_log_probs = self.replay_buff.sample(self.batch_size)
            # 策略更新
            self.ppo.update(
                states=states,
                actions=actions,
                rewards=rewards,
                dones=dones,
                old_log_probs=old_log_probs,
                episode=episode)

            if (episode+1) % 10000 == 0:
                # 保存模型
                model_name = os.path.join(self.save_path, "{}_ppo_{}w.pth".format(self.env_name[:-3], (episode+1) // 10000))
                torch.save(self.agent.state_dict(), model_name)
        self.env.close()
        self.writer.close()
        self.ppo.writer.close()



if __name__ == "__main__":
    config_loader = ConfigLoader("tools/config.yaml")
    config = config_loader.load_config()

    trainer = PPOTrainer(configs=config)
    trainer.train()