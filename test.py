import torch
import time
import numpy as np
import keyboard  # 导入keyboard库
from agent.agent import PPOAgent, SACAgent
from tools.config import ConfigLoader
from envs.unitreego2 import UnitreeGo2Env
from envs.panda import PandaEnv
from envs.piper import PiperEnv

if __name__ == "__main__":
    algorithm = "SAC"
    # load_path = "models/panda_sac_9w_old.pth"
    load_path = "models/piper_best_mode.pth"
    config_loader = ConfigLoader("tools/config_piper.yaml")
    configs = config_loader.load_config()
    robot_path = configs["environment"]["robot_path"]

    env = PiperEnv(robot_path)
    state_dim = env.get_observation_dim()
    action_dim = env.get_action_dim()
    if algorithm is "SAC":
        agent = SACAgent(state_dim, action_dim, configs["agent_network"]["hdim"])
    else:
        agent = PPOAgent(state_dim, action_dim, configs["agent_network"])

    checkpoint = torch.load(load_path)
    agent.load_state_dict(checkpoint)

    state = env.reset()

    # 将state声明为全局变量，以便在reset_env函数中修改
    global state_global
    state_global = state


    def reset_env():
        global state_global
        print("\nResetting environment...")
        state_global = env.reset()
        print("Environment reset complete!")


    # 注册热键 - 按 'r' 键重置环境
    keyboard.add_hotkey('r', reset_env)
    print("Press 'r' to reset environment. Press 'Ctrl+C' to exit.")

    while True:
        with torch.no_grad():
            # 使用全局变量state_global
            action = agent.get_action(state_global, deterministic=True)
        next_state, raw_reward, done, truncated, info = env.step(action)
        state_global = next_state
        env.render()

        # 检查是否需要重置环境（如果环境自己标记为done）
        if done:
            reset_env()

        time.sleep(0.02)