import mujoco
import mujoco.viewer
import numpy as np
import glfw
from typing import Dict, Tuple, Optional, Any


class PiperEnv:
    def __init__(self, xml_path: str = "scene1.xml", render: bool = False):
        """
        Piper 机械臂强化学习环境（适配 scene1.xml）

        参数:
            xml_path: 场景XML文件路径
            render: 是否开启渲染
        """
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(filename=xml_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 4

        # 获取站点ID - 使用 scene1.xml 中的站点名称
        self.fingertip_1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "finger7_site")
        self.fingertip_2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "finger8_site")

        # 获取传感器ID - 使用 scene1.xml 中的传感器名称
        self.finger1_contact_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "left_finger_contact")
        self.finger2_contact_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "right_finger_contact")

        # 检查传感器是否存在
        if self.finger1_contact_id == -1 or self.finger2_contact_id == -1:
            print("警告: 未找到接触传感器，请检查XML文件中的传感器定义")

        # 环境参数
        self.viewer = None
        self.max_steps = 500
        self.current_step = 0

        # 动作空间: 6个关节 + 1个夹爪
        self.action_dim = 7
        self.obs_dim = 27
        self.action_space = self._get_action_space()
        self.object_target_dis_init = None
        self.gripper_object_dis_init = None

        # 目标位置 - 使用 scene1.xml 中 target_area 的位置
        self.target_pos = np.array([0.6, 0, 0.035])  # 在目标区域上方一点

        # 重置环境
        self.reset()

    def get_observation_dim(self):
        return self.obs_dim

    def get_action_dim(self):
        return self.action_dim

    def _get_action_space(self) -> Dict[str, Any]:
        """定义动作空间 - 适配 Piper 机械臂的关节限位"""
        # Piper 机械臂的关节限位（根据 scene1.xml 中的 range 属性）
        joint_limits = [
            [-2.618, 2.618],  # joint1
            [0, 3.14],  # joint2
            [-2.697, 0],  # joint3
            [-1.832, 1.832],  # joint4
            [-1.22, 1.22],  # joint5
            [-3.14, 3.14],  # joint6
        ]

        # 夹爪限位 [0, 0.035]（joint7的范围）
        gripper_limit = [0, 0.035]

        return {
            'type': 'continuous',
            'shape': (self.action_dim,),
            'low': np.array([lim[0] for lim in joint_limits] + [gripper_limit[0]]),
            'high': np.array([lim[1] for lim in joint_limits] + [gripper_limit[1]])
        }

    def reset(self) -> np.ndarray:
        """重置环境"""
        mujoco.mj_resetData(self.model, self.data)

        # 设置物体的随机初始位置（在工作台范围内）
        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "training_object")
        if object_body_id == -1:
            raise ValueError("Body 'training_object' not found in model.")

        # 在工作台范围内随机生成物体位置
        table_pos = np.array([0, 0, -0.02])  # 工作台中心
        table_size = np.array([1, 1, 0.02])  # 工作台尺寸

        # 在工作台表面随机位置（避免边缘）
        x_range = [table_pos[0] - table_size[0] * 0.5, table_pos[0] + table_size[0] * 0.5]
        y_range = [table_pos[1] - table_size[1] * 0.5, table_pos[1] + table_size[1] * 0.5]

        random_pos = np.array([
            np.random.uniform(x_range[0], x_range[1]),
            np.random.uniform(y_range[0], y_range[1]),
            0.05  # 在工作台上方
        ])

        # 设置物体位置和方向
        self.data.qpos[8:11] = random_pos  # 物体位置
        self.data.qpos[11:15] = [1, 0, 0, 0]  # 单位四元数

        # 重置所有速度为零
        self.data.qvel[:] = 0.0

        # 前向动力学
        mujoco.mj_forward(self.model, self.data)

        # 计算初始距离
        self.object_target_dis_init = np.linalg.norm(random_pos - self.target_pos)

        # 获取指尖位置计算夹爪中心
        left_fingertip_pos = self.data.site_xpos[self.fingertip_1_id]
        right_fingertip_pos = self.data.site_xpos[self.fingertip_2_id]
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

        self.current_step += 1
        if self.current_step >= self.max_steps:
            terminated = True

        return observation, reward, terminated, info, reward_info

    def normalize_joint_pos(self, joint_pos):
        """将关节角度归一化到 [-1, 1] 范围"""
        joint_limits = np.array([
            [-2.618, 2.618],  # joint1
            [0, 3.14],  # joint2
            [-2.697, 0],  # joint3
            [-1.832, 1.832],  # joint4
            [-1.22, 1.22],  # joint5
            [-3.14, 3.14],  # joint6
        ])

        normalized = np.zeros_like(joint_pos)
        for i in range(6):
            min_val, max_val = joint_limits[i]
            if max_val > min_val:  # 避免除零
                normalized[i] = 2 * (joint_pos[i] - min_val) / (max_val - min_val) - 1
            else:
                normalized[i] = 0

        return normalized

    def normalize_joint_vel(self, joint_vel):
        """将关节速度归一化到 [-1, 1] 范围"""
        max_velocity = 2.0
        normalized = joint_vel / max_velocity
        normalized = np.clip(normalized, -1, 1)
        return normalized

    def normalize_gripper(self, gripper_pos):
        """将夹爪值归一化到 [-1, 1] 范围"""
        # 从 [0, 0.035] 映射到 [-1, 1]
        if gripper_pos > 0.035:
            gripper_pos = 0.035
        normalized = 2 * (gripper_pos / 0.035) - 1
        return np.array([normalized])

    def denormalize_action(self, action):
        """将 SAC 输出的动作 [-1, 1] 映射到实际控制范围"""
        joint_limits = np.array([
            [-2.618, 2.618],  # joint1
            [0, 3.14],  # joint2
            [-2.697, 0],  # joint3
            [-1.832, 1.832],  # joint4
            [-1.22, 1.22],  # joint5
            [-3.14, 3.14],  # joint6
            [0, 0.035]  # joint7 (夹爪)
        ])

        denormalized_action = np.zeros_like(action)
        for i in range(6):
            min_val, max_val = joint_limits[i]
            denormalized_action[i] = min_val + (action[i] + 1) * (max_val - min_val) / 2

        # 夹爪动作 (从 [-1, 1] 映射到 [0, 0.035])
        denormalized_action[6] = (action[6] + 1) * 0.035 / 2

        return denormalized_action

    def _set_action(self, action: np.ndarray):
        """设置控制信号 - 适配 scene1.xml 的执行器命名"""
        action = self.denormalize_action(action)

        # 前6个是关节控制
        for i in range(6):
            actuator_name = f"joint{i + 1}"
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            if actuator_id != -1:
                self.data.ctrl[actuator_id] = action[i]

        # 第7个是夹爪控制
        gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        if gripper_id != -1:
            self.data.ctrl[gripper_id] = action[6]

    def normalize_touch(self, touch):
        """将接触力归一化到 [0, 1] 范围"""
        normalized = np.clip(touch, 0, 1)
        return normalized

    def _get_observation(self) -> np.ndarray:
        """获取观察 - 保持与 PandaEnv 相同的观察结构"""
        # 关节位置 (6个)
        joint_pos = self.data.qpos[0:6]

        # 关节速度 (6个)
        joint_vel = self.data.qvel[0:6]

        # 夹爪开合程度 (1个) - joint7的位置，qpos[7]是夹爪的另一半[-0.035,0]
        gripper_aperture = self.data.qpos[6]

        # 左指尖位置 (3维)
        left_fingertip_pos = self.data.site_xpos[self.fingertip_1_id]

        # 右指尖位置 (3维)
        right_fingertip_pos = self.data.site_xpos[self.fingertip_2_id]

        # 物体位置 (3个)
        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "training_object")
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

        # 组合所有观察6+6+1+3+3+3+3+2=27
        observation = np.concatenate([
            norm_joint_pos,  # 6
            norm_joint_vel,  # 6
            norm_gripper_aperture,  # 1
            left_fingertip_pos,  # 3
            right_fingertip_pos,  # 3
            object_pos,  # 3
            target_pos,  # 3
            [norm_touch1, norm_touch2]  # 2
        ])

        return observation

    def _get_reward(self) -> Tuple[float, Dict]:
        """计算综合奖励（与 PandaEnv 相同的奖励逻辑）"""
        reward_info = {}

        # 获取关键位置信息
        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "training_object")
        object_pos = self.data.xpos[object_body_id]

        # 获取指尖位置
        left_fingertip_pos = self.data.site_xpos[self.fingertip_1_id]
        right_fingertip_pos = self.data.site_xpos[self.fingertip_2_id]
        gripper_center_pos = (left_fingertip_pos + right_fingertip_pos) / 2.0

        # 计算距离
        gripper_to_object_dist = np.linalg.norm(gripper_center_pos - object_pos)
        object_to_target_dist = np.linalg.norm(object_pos - self.target_pos)

        # 判断接触状态
        touch1 = self.data.sensordata[self.finger1_contact_id] if self.finger1_contact_id != -1 else 0.0
        touch2 = self.data.sensordata[self.finger2_contact_id] if self.finger2_contact_id != -1 else 0.0
        graspable_ok = (touch1 > 0 and touch2 > 0) and gripper_to_object_dist < 0.015
        bonus_ok = object_to_target_dist < 0.025

        # 计算各奖励分量（与 PandaEnv 相同的逻辑）
        # 接近奖励
        if self.gripper_object_dis_init > 0:
            ratio = (gripper_to_object_dist - self.gripper_object_dis_init) / self.gripper_object_dis_init
            approach_reward = -ratio * 2
        else:
            approach_reward = 0
        reward_info['approach_reward'] = approach_reward

        # 接触奖励
        touch_reward = 10 if (graspable_ok and not bonus_ok) else 0
        reward_info['touch_reward'] = touch_reward

        # 放置奖励
        if graspable_ok and self.object_target_dis_init > 0:
            ratio = (object_to_target_dist - self.object_target_dis_init) / self.object_target_dis_init
            placement_reward = -ratio * 2
        else:
            placement_reward = 0
        reward_info['placement_reward'] = placement_reward

        # 成功奖励
        success_bonus = 200 if bonus_ok else 0
        reward_info['success_bonus'] = success_bonus

        # 总奖励
        total_reward = (
                approach_reward +
                touch_reward +
                placement_reward +
                success_bonus
        )

        return total_reward, reward_info

    def _is_terminated(self) -> bool:
        """检查是否终止"""
        object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "training_object")
        object_pos = self.data.xpos[object_body_id]

        # 检查物体是否掉落
        if object_pos[2] < 0:
            return True

        # 检查是否成功
        distance = np.linalg.norm(object_pos - self.target_pos)
        if distance < 0.05:
            return True

        return False

    def render(self):
        """渲染环境"""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer.sync()

    def close(self):
        """关闭环境"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    # 测试环境
    env = PiperEnv("piper/scene1.xml")
    obs = env.reset()
    print(f"观察维度: {len(obs)}")
    print(f"观察: {obs}")

    # 测试一步动作
    action = np.random.uniform(-1, 1, env.get_action_dim())
    obs, reward, terminated, info, reward_info = env.step(action)
    print(f"奖励: {reward}")
    print(f"终止: {terminated}")

    env.close()