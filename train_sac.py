import os.path
import torch
import time
import numpy as np

from agent.agent import  SACAgent
from algorithm.sac import SAC
from algorithm.replay_buffer import ReplayBuffer, DualExperienceReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from tools.config import ConfigLoader
from envs.piper import PiperEnv
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACTrainer:
    def __init__(self, configs: dict):
        # 初始化环境
        self.render = configs["training"]["render"]
        self.env_name = configs["environment"]["name"]
        self.robot_path = configs["environment"]["robot_path"]
        self.pre_model_path = configs["training"]["pre_model_path"]
        self.save_path = configs["training"]["output_dir"]

        # 加载环境
        self.env = PiperEnv(self.robot_path)
        self.state_dim = self.env.get_observation_dim()
        self.action_dim = self.env.get_action_dim()
        self.agent_network = configs["agent_network"]["hdim"]

        # 初始化Agent
        self.agent = SACAgent(self.state_dim, self.action_dim, hidden_dim=self.agent_network).to(device)

        if self.pre_model_path:
            checkpoint = torch.load(self.pre_model_path, map_location=device)
            self.agent.load_state_dict(checkpoint)

        # 实例化算法
        self.sac = SAC(agent=self.agent)

        # 训练参数
        self.max_episodes = configs["training"].get('max_episodes', 100000)
        self.max_steps = configs["training"].get('max_steps', 200)
        self.batch_size = configs['training']['batch_size']
        self.buffer_size = configs['algorithm_sac']['buffer_size']

        # 初始化经验池
        self.replay_buff = ReplayBuffer(capacity=self.buffer_size)
        # self.replay_buff = DualExperienceReplayBuffer(total_capacity=self.buffer_size, grasp_ratio=0.2)

        self.writer = SummaryWriter(f'runs/{self.env_name}_sac_{int(time.time())}')
        self.max_reward = 0
    def train(self):
        """主训练循环"""
        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = []
            reward_infos = []
            loss_infos = []

            for i in range(self.max_steps):
                action = self.agent.get_action(state)
                next_state, reward, done, info, reward_info = self.env.step(action)
                self.replay_buff.add(state, action, reward, next_state, done)

                state = next_state
                episode_reward.append(reward)
                reward_info_list = list(reward_info.values())
                reward_infos.append(reward_info_list)

                if self.replay_buff.size > self.batch_size:
                    batch = self.replay_buff.sample(self.batch_size)
                    loss_info = self.sac.update(batch)

                    loss_infos.append(list(loss_info.values()))

                if self.render:
                    self.env.render()

                if done:
                    break

            episode_reward_mean = sum(episode_reward) / len(episode_reward)
            reward_infos = sum(np.array(reward_infos)) / len(episode_reward)
            loss = sum(np.array(loss_infos)) / len(loss_infos)

            print(f"\rEpisode {episode + 1}/{self.max_episodes}, Total Reward: {episode_reward_mean:.2f}, "
                  f"approach_reward: {reward_infos[0]:.3f}, touch_reward: {reward_infos[1]:.3f}, "
                  f"placement_reward: {reward_infos[2]:.3f}, success_bonus: {reward_infos[3]:.3f},",
                  end="", flush=True)

            self.writer.add_scalar('Episode_Reward', episode_reward_mean, episode)

            self.writer.add_scalar('approach_reward', reward_infos[0], episode)
            self.writer.add_scalar('touch_reward', reward_infos[1], episode)
            self.writer.add_scalar('placement_reward', reward_infos[2], episode)
            self.writer.add_scalar('success_bonus', reward_infos[3], episode)

            self.writer.add_scalar("q1_loss", loss[0], episode)
            self.writer.add_scalar("q2_loss", loss[1], episode)
            self.writer.add_scalar("policy_loss", loss[2], episode)
            self.writer.add_scalar("alpha_loss", loss[3], episode)

            # 保存模型
            if self.max_reward < episode_reward_mean:
                self.max_reward = episode_reward_mean
                best_model_name = os.path.join(self.save_path, f"{self.env_name}_best_mode.pth")
                torch.save(self.agent.state_dict(), best_model_name)
            if (episode+1) % 10000 == 0:
                model_name = os.path.join(self.save_path, "{}_sac_{}w.pth".format(self.env_name, (episode+1) // 10000))
                torch.save(self.agent.state_dict(), model_name)

        self.env.close()
        self.writer.close()


if __name__ == "__main__":
    config_loader = ConfigLoader("tools/config_piper.yaml")
    config = config_loader.load_config()

    trainer = SACTrainer(configs=config)
    trainer.train()