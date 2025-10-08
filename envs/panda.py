import mujoco
import mujoco.viewer
import numpy as np
import glfw
from typing import Dict, Tuple, Optional, Any


class PandaEnv:
    def __init__(self, xml_path: str = "scene1.xml", render: bool = False):
        """
        Panda 机械臂强化学习环境

        参数:
            xml_path: 场景XML文件路径
            render: 是否开启渲染
        """
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 4

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
        self.viewer = None
        self.max_steps = 500
        self.current_step = 0

        # 动作空间: 7个关节 + 1个夹爪
        self.action_dim = 8
        self.obs_dim = 29
        self.action_space = self._get_action_space()
        self.object_target_dis_init = None
        self.gripper_object_dis_init = None

        # 目标位置
        self.target_pos = np.array([0.0, 0.65, 0.025])

        # 重置环境
        self.reset()
    def get_observation_dim(self):
        return self.obs_dim

    def get_action_dim(self):
        return self.action_dim

    def _get_action_space(self) -> Dict[str, Any]:
        """定义动作空间"""
        # 7个关节的限位
        joint_limits = [
            [-2.8973, 2.8973],  # joint1
            [-1.7628, 1.7628],  # joint2
            [-2.8973, 2.8973],  # joint3
            [-3.0718, -0.0698],  # joint4
            [-2.8973, 2.8973],  # joint5
            [-0.0175, 3.7525],  # joint6
            [-2.8973, 2.8973]  # joint7
        ]

        # 夹爪限位 [0, 255]
        gripper_limit = [0, 255]

        return {
            'type': 'continuous',
            'shape': (self.action_dim,),
            'low': np.array([lim[0] for lim in joint_limits] + [gripper_limit[0]]),
            'high': np.array([lim[1] for lim in joint_limits] + [gripper_limit[1]])
        }

    def reset(self) -> np.ndarray:
        mujoco.mj_resetData(self.model, self.data)

        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "graspable_object")
        if object_body_id == -1:
            raise ValueError("Body 'graspable_object' not found in model.")

        radius = np.random.uniform(0.3, 0.55, 2)
        # 2. 随机选择象限 (0-3)
        quadrant = np.random.randint(0, 4)

        # 3. 根据象限确定坐标符号
        if quadrant == 0:  # 第一象限 (x+, y+)
            x_sign, y_sign = 1, 1
        elif quadrant == 1:  # 第二象限 (x-, y+)
            x_sign, y_sign = -1, 1
        elif quadrant == 2:  # 第三象限 (x-, y-)
            x_sign, y_sign = -1, -1
        else:  # 第四象限 (x+, y-)
            x_sign, y_sign = 1, -1

        # 4. 生成随机坐标
        random_pos = np.array([x_sign * radius[0], y_sign * radius[1], 0.05])
        self.data.qpos[9:12] = random_pos  # 设置位置 (x, y, z)
        self.data.qpos[12:16] = [1, 0, 0, 0]  # 设置方向为单位四元数

        # 重置物体的速度为零
        self.data.qvel[:] = 0.0  # 重置所有速度为零

        # 前向动力学
        mujoco.mj_forward(self.model, self.data)

        # 计算初始位置与目标点的距离
        self.object_target_dis_init = np.linalg.norm(random_pos - self.target_pos)

        # 获取左右指尖位置
        left_fingertip_pos = self.data.site_xpos[self.fingertip_1_id]
        right_fingertip_pos = self.data.site_xpos[self.fingertip_2_id]
        # 计算指尖中心位置（近似夹爪中心）与目标物的初始距离
        gripper_center_pos = (left_fingertip_pos + right_fingertip_pos) / 2.0
        self.gripper_object_dis_init = np.linalg.norm(random_pos - gripper_center_pos)

        self.current_step = 0
        return self._get_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict, Dict]:
        """执行一步动作"""
        # 设置控制信号
        self._set_action(action)
        info = {}

        # 执行仿真步骤
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # 获取观察
        observation = self._get_observation()

        # 计算奖励
        reward, reward_info = self._get_reward()

        # 检查是否终止
        terminated = self._is_terminated()

        return observation, reward, terminated, info, reward_info

    def normalize_joint_pos(self, joint_pos):
        """将关节角度归一化到 [-1, 1] 范围"""
        joint_limits = np.array([
            [-2.8973, 2.8973],  # joint1
            [-1.7628, 1.7628],  # joint2
            [-2.8973, 2.8973],  # joint3
            [-3.0718, -0.0698],  # joint4
            [-2.8973, 2.8973],  # joint5
            [-0.0175, 3.7525],  # joint6
            [-2.8973, 2.8973]  # joint7
        ])

        # 最小-最大归一化: (x - min) / (max - min) * 2 - 1
        normalized = np.zeros_like(joint_pos)
        for i in range(7):
            min_val, max_val = joint_limits[i]
            normalized[i] = 2 * (joint_pos[i] - min_val) / (max_val - min_val) - 1

        return normalized

    def normalize_joint_vel(self, joint_vel):
        """将关节速度归一化到 [-1, 1] 范围"""
        max_velocity = 2.0  # 假设最大速度为 2.0 rad/s
        normalized = joint_vel / max_velocity
        # 确保值在 [-1, 1] 范围内
        normalized = np.clip(normalized, -1, 1)
        return normalized

    def normalize_gripper(self, gripper_pos):
        """将夹爪值归一化到 [-1, 1] 范围"""
        # 从 [0, 255] 映射到 [-1, 1]
        normalized = 2 * (gripper_pos / 255) - 1
        return normalized

    def denormalize_action(self, action):
        """将 SAC 输出的动作 [-1, 1] 映射到实际控制范围"""
        joint_limits = np.array([
            [-2.8973, 2.8973],  # joint1
            [-1.7628, 1.7628],  # joint2
            [-2.8973, 2.8973],  # joint3
            [-3.0718, -0.0698],  # joint4
            [-2.8973, 2.8973],  # joint5
            [-0.0175, 3.7525],  # joint6
            [-2.8973, 2.8973]  # joint7
        ])
        # 前7个关节动作
        denormalized_action = np.zeros_like(action)
        for i in range(7):
            min_val, max_val = joint_limits[i]
            # 从 [-1, 1] 映射到 [min_val, max_val]
            denormalized_action[i] = min_val + (action[i] + 1) * (max_val - min_val) / 2

        # 夹爪动作 (从 [-1, 1] 映射到 [0, 255])
        denormalized_action[7] = (action[7] + 1) * 255 / 2

        return denormalized_action

    def _set_action(self, action: np.ndarray):
        """设置控制信号"""
        action = self.denormalize_action(action)
        # 前7个是关节控制
        for i in range(7):
            actuator_name = f"actuator{i + 1}"
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            self.data.ctrl[actuator_id] = action[i]

        # 第8个是夹爪控制
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.data.ctrl[gripper_id] = action[7]

    def normalize_touch(self, touch):
        """将位置坐标归一化到 [-1, 1] 范围"""
        # 假设位置坐标在 [-1, 1] 米范围内
        normalized = np.clip(touch, 0, 2)
        return normalized

    def _get_observation(self) -> np.ndarray:
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
        object_pos = self.data.xpos[object_body_id]

        # 目标位置 (3个)
        target_pos = self.target_pos

        # 接触传感器 (2个)
        touch1 = self.data.sensordata[self.finger1_contact_id] if self.finger1_contact_id != -1 else 0.0
        touch2 = self.data.sensordata[self.finger2_contact_id] if self.finger2_contact_id != -1 else 0.0

        # 归一化各个部分
        norm_joint_pos = self.normalize_joint_pos(joint_pos)
        norm_joint_vel = self.normalize_joint_vel(joint_vel)
        norm_gripper_aperture = self.normalize_gripper(gripper_aperture)
        norm_touch1 = self.normalize_touch(touch1)
        norm_touch2 = self.normalize_touch(touch2)

        # 组合所有观察
        observation = np.concatenate([
            norm_joint_pos,
            norm_joint_vel,
            norm_gripper_aperture,
            left_fingertip_pos,
            right_fingertip_pos,
            object_pos,
            target_pos,
            [norm_touch1, norm_touch2]
        ])

        return observation

    def _get_reward(self) -> Tuple[float, Dict]:
        """
        计算综合奖励，由多个部分组成
        返回: (总奖励, 奖励信息字典)
        """
        reward_info = {}  # 存储各奖励分量的详细信息

        # 1. 获取关键位置信息
        # object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "graspable_object")
        object_pos = self.data.qpos[9:12]

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

        # 2.1 计算指尖与目标物距离奖励，越靠近奖励越大[-1, 1]
        if self.gripper_object_dis_init == 0:
            # 避免除以零
            ratio = 0
        else:
            ratio = (gripper_to_object_dist - self.gripper_object_dis_init) / self.gripper_object_dis_init
        # approach_reward = -np.tanh(ratio * 2)
        approach_reward = -ratio * 2
        reward_info['approach_reward'] = approach_reward

        # 2.2接触奖励（夹取奖励）
        if graspable_ok and not bonus_ok:
            touch_reward = 10
        else:
            touch_reward = 0
        reward_info['touch_reward'] = touch_reward

        # 2.3 在物体被夹起后靠近目标区域奖励越高
        if graspable_ok:
            if self.object_target_dis_init == 0:
                # 避免除以零
                ratio = 0
            else:
                ratio = (object_to_target_dist - self.object_target_dis_init) / self.object_target_dis_init
            # placement_reward = -np.tanh(ratio) * 2
            placement_reward = -ratio * 2
        else:
            placement_reward = 0
        reward_info['placement_reward'] = placement_reward

        # 2.4 成功放置奖励
        if bonus_ok:
            success_bonus = 200
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

    def _is_terminated(self) -> bool:
        """检查是否终止"""
        # 获取物体位置
        # object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "graspable_object")
        object_pos = self.data.qpos[9:12]

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

    def render(self):
        """
        渲染环境
        """
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

    def close(self):
        """关闭环境并释放资源"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    env = PandaEnv("panda/scene.xml")
    obs = env._get_observation()
    print(obs)
    print(len(obs))


