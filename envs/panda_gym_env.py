import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import glfw
from typing import Dict, Tuple, Optional, Any
import os


class PandaGymEnv(gym.Env):
    """符合 Gymnasium 接口的 Panda 机械臂环境"""

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, xml_path="scene1.xml", render_mode=None):
        super(PandaGymEnv, self).__init__()

        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 4

        # 渲染模式
        self.render_mode = render_mode
        self.viewer = None

        # 获取站点ID
        self.fingertip_1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fingertip_1")
        self.fingertip_2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "fingertip_2")

        # 获取传感器ID
        self.finger1_contact_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "finger1_contact")
        self.finger2_contact_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "finger2_contact")

        # 检查传感器是否存在
        if self.finger1_contact_id == -1 or self.finger2_contact_id == -1:
            print("警告: 未找到接触传感器，请检查XML文件中的传感器定义")

        # 环境参数
        self.max_steps = 500
        self.current_step = 0

        # 动作空间: 7个关节 + 1个夹爪
        self.action_dim = 8

        # 定义动作空间
        joint_limits = [
            [-2.8973, 2.8973],  # joint1
            [-1.7628, 1.7628],  # joint2
            [-2.8973, 2.8973],  # joint3
            [-3.0718, -0.0698],  # joint4
            [-2.8973, 2.8973],  # joint5
            [-0.0175, 3.7525],  # joint6
            [-2.8973, 2.8973]  # joint7
        ]

        # 夹爪限位 [0, 0.04] 映射到 [0, 255]
        gripper_limit = [0, 255]

        self.action_space = spaces.Box(
            low=np.array([lim[0] for lim in joint_limits] + [gripper_limit[0]]),
            high=np.array([lim[1] for lim in joint_limits] + [gripper_limit[1]]),
            dtype=np.float32
        )

        # 观察空间维度
        self.obs_dim = 29  # 根据_get_observation方法计算

        # 定义观察空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # 目标位置
        self.target_pos = np.array([0.7, -0.3, 0.025])

        # 重置环境
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # 设置物体初始位置
        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "graspable_object")
        if object_body_id != -1:
            self.data.xpos[object_body_id] = np.array([0.5, 0.5, 0.05])

        # 重置物体的速度为零
        self.data.qvel[:] = 0.0

        # 前向动力学
        mujoco.mj_forward(self.model, self.data)

        self.current_step = 0

        # 获取观察
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        # 设置控制信号
        self._set_action(action)

        # 执行仿真步骤
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # 获取观察
        observation = self._get_observation()

        # 计算奖励
        reward, reward_info = self._get_reward()

        # 检查是否终止
        terminated = self._is_terminated()

        # 检查是否截断（达到最大步数）
        self.current_step += 1
        truncated = self.current_step >= self.max_steps

        # 信息字典
        info = reward_info
        info['is_success'] = self._is_success()

        return observation, reward, terminated, truncated, info

    def _set_action(self, action):
        """设置控制信号"""
        # 前7个是关节控制
        for i in range(7):
            actuator_name = f"actuator{i + 1}"
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            self.data.ctrl[actuator_id] = action[i]

        # 第8个是夹爪控制
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.data.ctrl[gripper_id] = action[7]

    def _get_observation(self):
        """获取观察"""
        # 关节位置 (7个)
        joint_pos = self.data.qpos[0:7]

        # 关节速度 (7个)
        joint_vel = self.data.qvel[0:7]

        # 夹爪开合程度 (1个)
        gripper_aperture = self.data.qpos[7:8]

        # 左指尖位置 (3维)
        left_fingertip_pos = self.data.site_xpos[self.fingertip_1_id]

        # 右指尖位置 (3维)
        right_fingertip_pos = self.data.site_xpos[self.fingertip_2_id]

        # 物体位置 (3个)
        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "graspable_object")
        object_pos = self.data.xpos[object_body_id] if object_body_id != -1 else np.zeros(3)

        # 目标位置 (3个)
        target_pos = self.target_pos

        # 接触传感器 (2个)
        touch1 = self.data.sensordata[self.finger1_contact_id] if self.finger1_contact_id != -1 else 0.0
        touch2 = self.data.sensordata[self.finger2_contact_id] if self.finger2_contact_id != -1 else 0.0

        # 组合所有观察
        observation = np.concatenate([
            joint_pos,
            joint_vel,
            gripper_aperture,
            left_fingertip_pos,
            right_fingertip_pos,
            object_pos,
            target_pos,
            [touch1, touch2]
        ])

        return observation

    def _get_reward(self):
        """
        计算综合奖励，由多个部分组成
        返回: (总奖励, 奖励信息字典)
        """
        reward_info = {}  # 存储各奖励分量的详细信息

        # 1. 获取关键位置信息
        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "graspable_object")
        object_pos = self.data.xpos[object_body_id] if object_body_id != -1 else np.zeros(3)

        # 获取左右指尖位置
        left_fingertip_pos = self.data.site_xpos[self.fingertip_1_id]
        right_fingertip_pos = self.data.site_xpos[self.fingertip_2_id]

        # 计算指尖中心位置（近似夹爪中心）
        gripper_center_pos = (left_fingertip_pos + right_fingertip_pos) / 2.0

        # 目标位置
        target_pos = self.target_pos

        # 计算指尖与目标物的距离、计算目标物与指定区域的距离
        gripper_to_object_dist = np.linalg.norm(gripper_center_pos - object_pos)
        object_to_target_dist = np.linalg.norm(object_pos - target_pos)

        # 判断是否抓取目标物
        touch1 = self.data.sensordata[self.finger1_contact_id] if self.finger1_contact_id != -1 else 0.0
        touch2 = self.data.sensordata[self.finger2_contact_id] if self.finger2_contact_id != -1 else 0.0
        graspable_ok = (touch1 > 0 and touch2 > 0) and gripper_to_object_dist < 0.05

        # 判断目标物是否放入指定区域
        bonus_ok = object_to_target_dist < 0.05

        # 2.1 计算指尖与目标物距离奖励，使用指数衰减将距离映射到[0,1]范围
        approach_reward = np.exp(-gripper_to_object_dist / 0.1)
        reward_info['approach_reward'] = approach_reward

        # 2.2接触奖励（夹取奖励）
        if graspable_ok and not bonus_ok:
            touch_reward = 3
        else:
            touch_reward = 0
        reward_info['touch_reward'] = touch_reward

        # 2.3 目标物与目标区域的距离奖励 [0, 1]
        placement_reward = np.exp(-object_to_target_dist / 0.1)
        reward_info['placement_reward'] = placement_reward

        # 2.4 成功放置奖励
        if bonus_ok:
            success_bonus = 10
        else:
            success_bonus = 0
        reward_info['success_bonus'] = success_bonus

        # 3. 组合总奖励（可以调整权重）
        total_reward = (
                approach_reward +  # 引导接近物体
                touch_reward * 2.0 +  # 鼓励抓取（加权更高）
                placement_reward +  # 引导移动到目标区域
                success_bonus  # 成功放置的奖励
        )

        return total_reward, reward_info

    def _is_terminated(self):
        """检查是否终止"""
        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "graspable_object")
        if object_body_id == -1:
            return False

        object_pos = self.data.xpos[object_body_id]

        # 检查物体是否掉落到地面以下
        if object_pos[2] < 0:
            return True

        # 检查是否成功
        target_pos = self.target_pos
        distance = np.linalg.norm(object_pos - target_pos)
        success_threshold = 0.05

        if distance < success_threshold:
            return True

        return False

    def _is_success(self):
        """检查是否成功完成任务"""
        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "graspable_object")
        if object_body_id == -1:
            return False

        object_pos = self.data.xpos[object_body_id]
        target_pos = self.target_pos
        distance = np.linalg.norm(object_pos - target_pos)

        return distance < 0.05

    def render(self):
        """渲染环境"""
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            else:
                self.viewer.sync()

    def close(self):
        """关闭环境并释放资源"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None