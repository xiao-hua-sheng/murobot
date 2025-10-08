import os
import time
import numpy as np
import mujoco
import mujoco.viewer
from typing import Optional, Dict, Any, Tuple, List


class UnitreeGo2Env:
    """
    使用纯MuJoCo API的Unitree Go2环境
    避免使用gymnasium/spaces，直接处理观察空间维度
    """

    def __init__(self, model_path: str, frame_skip: int = 1):
        """
        初始化Go2环境

        参数:
            model_path: MJCF模型文件路径
            frame_skip: 每次动作执行的仿真步数
        """
        # 加载MuJoCo模型
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.reward_weight = {
            "movement_reward": 1,
            "height_reward": 1,
            "action_constraint_reward": 0
        }

        # 初始化可视化器
        self.viewer = None

        # 分析模型结构，获取正确的观察空间维度
        self.obs_dim = self._calculate_obs_dim()
        print(f"Calculated observation dimension: {self.obs_dim}")

        # 动作空间维度（12个关节）
        self.act_dim = 12  # Unitree Go2有12个可控关节

        # 重置环境
        self.reset()

    def _calculate_obs_dim(self) -> int:
        """
        计算观察空间的真实维度
        基于MuJoCo模型的实际数据结构:cite[5]
        """
        # 基本组件：关节位置 + 关节速度
        dim = self.model.nq + self.model.nv

        # 添加执行器力信息
        dim += self.model.nu

        # 添加身体惯性量和速度信息
        dim += 10  # 简化处理，实际可能需要根据模型调整

        print(f"Model nq: {self.model.nq}, nv: {self.model.nv}, nu: {self.model.nu}")
        return dim

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        重置环境到初始状态

        返回:
            观察值数组
        """
        # 重置MuJoCo模型和数据
        mujoco.mj_resetData(self.model, self.data)

        # 设置初始关节位置（可选）
        # 这里可以设置机器狗的初始姿态

        # 向前仿真一步以确保所有量已更新
        mujoco.mj_forward(self.model, self.data)

        # 获取初始观察值
        observation = self._get_obs()

        # 验证观察值维度
        if len(observation) != self.obs_dim:
            print(f"Warning: Expected dim {self.obs_dim}, got {len(observation)}. Updating obs_dim.")
            self.obs_dim = len(observation)

        return observation

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any], List]:
        """
        执行一个动作

        参数:
            action: 要执行的动作
        返回:
            observation: 新的观察值
            reward: 奖励值
            terminated: 是否终止
            info: 信息字典
        """
        # 应用动作
        self._apply_action(action)

        # 执行仿真步骤
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # 获取新的观察值
        observation = self._get_obs()

        # 计算奖励
        reward, reward_info  = self._calculate_reward()

        # 检查是否终止
        terminated = self._check_terminated()

        # 获取信息
        info = self._get_info()

        return observation, reward, terminated, info, reward_info

    def _apply_action(self, action: np.ndarray):
        """
        将动作应用到模型
        """
        # 确保动作维度正确
        if len(action) != self.act_dim:
            raise ValueError(f"Action dimension mismatch. Expected {self.act_dim}, got {len(action)}")

        # 将归一化动作转换为实际控制信号
        max_torque = 20.0  # 最大扭矩值，需要根据Go2的实际参数调整
        self.data.ctrl[:] = action * max_torque

    def _get_obs(self) -> np.ndarray:
        """
        获取当前观察值，确保维度一致性:cite[3]:cite[5]
        """
        # 确保所有组件都是一维数组:cite[3]
        qpos = self.data.qpos.copy().flatten()
        qvel = self.data.qvel.copy().flatten()
        ctrl = self.data.ctrl.copy().flatten()

        # 获取身体姿态和位置
        body_quat = self.data.xquat[1].copy().flatten()  # 通常索引1是身体
        body_pos = self.data.xpos[1].copy().flatten()  # 身体位置

        # 组合观察值:cite[5]
        observation = np.concatenate([
            qpos,  # 所有位置
            qvel,  # 所有速度
            ctrl,  # 控制信号
            body_quat,  # 身体方向（四元数）
            body_pos  # 身体位置
        ]).astype(np.float32)

        return observation

    def _calculate_reward(self) :
        """
        计算Unitree Go2机器狗的奖励函数
        包含运动奖励、高度奖励和动作约束奖励三部分
        """
        # 1. 运动奖励 - 鼓励向前移动和保持适当的速度
        movement_reward = self.reward_weight["movement_reward"] * self._calculate_movement_reward()

        # 2. 高度奖励 - 鼓励保持适当的高度和稳定姿态
        height_reward = self.reward_weight["height_reward"] * self._calculate_height_reward()

        # 3. 动作约束奖励 - 惩罚过大或过快的动作变化
        action_constraint_reward = self.reward_weight["action_constraint_reward"] * self._calculate_action_constraint_reward()

        # 组合奖励（可根据需要调整权重）
        total_reward = movement_reward + height_reward + action_constraint_reward

        reward_info = [movement_reward, height_reward, action_constraint_reward]

        return float(total_reward), reward_info

    def _calculate_movement_reward(self) -> float:
        """
        计算运动相关奖励
        鼓励向前移动，惩罚后退和侧向移动
        """
        # 获取基座的速度（前3个是线速度，后3个是角速度）
        base_vel = self.data.qvel[:6].copy()

        # 前进速度奖励（x方向）
        forward_velocity = base_vel[0]
        forward_reward = forward_velocity * 1  # 缩放因子可根据需要调整

        # 侧向移动惩罚（y方向）
        lateral_velocity = base_vel[1]
        lateral_penalty = -abs(lateral_velocity) * 0.25

        # 旋转惩罚（鼓励直线前进）
        angular_velocity = np.linalg.norm(base_vel[3:6])
        rotation_penalty = -angular_velocity * 0.25

        # 总运动奖励
        movement_reward = forward_reward + lateral_penalty + rotation_penalty

        return movement_reward

    def _calculate_height_reward(self) -> float:
        """
        计算高度和姿态相关奖励
        鼓励保持适当高度和水平姿态
        """
        # 获取基座高度
        base_height = self.data.xpos[1, 2]

        # 目标高度（从模型文件中可以看出初始高度约为0.445）
        target_height = 0.4  # 略低于初始高度，更稳定的姿态

        # 高度奖励（高斯函数，在目标高度处奖励最大）
        height_diff = abs(base_height - target_height)
        height_reward = np.exp(-height_diff * 5.0) - 0.23 # 5.0是缩放因子

        # 姿态奖励（鼓励身体保持水平）
        # 获取身体z轴在世界坐标系中的方向
        z_axis = self.data.xmat[1][6:9]
        upness = z_axis[2]  # z轴在世界坐标系z方向的分量

        # 当身体直立时，upness接近1；倾斜时接近0
        orientation_reward = upness * 0.5

        # 总高度和姿态奖励
        height_reward_total = height_reward + orientation_reward

        return height_reward_total

    def _calculate_action_constraint_reward(self) -> float:
        """
        计算动作约束相关奖励
        对每个超出力矩限制的执行器给予固定惩罚，限定范围内奖励为0
        """
        # 初始化惩罚值
        total_penalty = 0.0
        fixed_penalty_per_actuator = -0.1  # 每个超出限制的执行器的固定惩罚值

        # 定义执行器力矩限制
        # 根据XML文件，abduction和hip电机的范围是[-23.7,23.7]，knee电机的范围是[-45.4,45.4]
        torque_limits = []
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if actuator_name and "calf" in actuator_name:  # 膝关节执行器
                torque_limits.append(45.4)
            else:  # 外展关节和髋关节执行器
                torque_limits.append(23.7)

        # 遍历所有执行器
        for i in range(self.model.nu):
            # 获取当前执行器的控制信号（力矩值）
            torque = self.data.ctrl[i]
            torque_limit = torque_limits[i]

            # 检查力矩是否超出限制
            if abs(torque) > torque_limit:
                total_penalty += fixed_penalty_per_actuator

        # 总动作约束奖励（惩罚）
        action_constraint_reward = total_penalty

        return action_constraint_reward

    def _check_terminated(self) -> bool:
        """
        检查是否终止回合
        当机器人跌倒或超出边界时终止
        """
        # 检查身体高度是否过低（跌倒）
        base_height = self.data.xpos[1, 2]
        if base_height < 0.2:  # 如果身体高度低于0.2米
            return True

        # 检查身体倾斜角度是否过大
        z_axis = self.data.xmat[1][6:9]  # 身体的z轴方向
        upness = z_axis[2]  # z轴在世界坐标系z方向的分量
        if upness < 0.5:  # 如果身体倾斜超过60度
            return True

        # 检查是否超出边界（可选）
        x_pos = self.data.xpos[1, 0]
        y_pos = self.data.xpos[1, 1]
        if abs(x_pos) > 10.0 or abs(y_pos) > 10.0:  # 10米边界
            return True

        return False


    def _get_info(self) -> Dict[str, Any]:
        """
        获取环境信息
        """
        info = {
            'x_position': self.data.xpos[1, 0],
            'y_position': self.data.xpos[1, 1],
            'z_position': self.data.xpos[1, 2],
            'survival_time': self.data.time,
            'distance': np.sqrt(self.data.xpos[1, 0] ** 2 + self.data.xpos[1, 1] ** 2),
            'observation_dim': self.obs_dim,
            'action_dim': self.act_dim
        }
        return info

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

    def get_observation_dim(self) -> int:
        """获取观察空间维度"""
        return self.obs_dim

    def get_action_dim(self) -> int:
        """获取动作空间维度"""
        return self.act_dim


# 训练函数
def train_go2(model_path: str, total_timesteps: int = 1000000):
    """
    训练Unitree Go2机器狗
    """
    # 创建环境
    env = UnitreeGo2Env(model_path)
    obs_dim = env.get_observation_dim()
    act_dim = env.get_action_dim()

    print(f"Observation dimension: {obs_dim}")
    print(f"Action dimension: {act_dim}")

    # 训练参数
    max_episode_steps = 1000
    save_interval = 10000

    # 训练循环
    observation = env.reset()
    episode_reward = 0
    episode_length = 0

    for step in range(total_timesteps):
        # 选择动作（替换为你的算法）
        action = np.random.uniform(-1, 1, size=act_dim)

        # 执行动作
        next_observation, reward, terminated, info = env.step(action)

        # 更新状态
        observation = next_observation
        episode_reward += reward
        episode_length += 1

        # 定期打印进度
        if step % 1000 == 0:
            print(f"Step: {step}, Reward: {reward:.2f}, Total Reward: {episode_reward:.2f}")

        # 检查回合是否结束
        if terminated or episode_length >= max_episode_steps:
            print(f"Episode finished after {episode_length} steps, reward: {episode_reward:.2f}")
            observation = env.reset()
            episode_reward = 0
            episode_length = 0

        # 定期保存模型
        if step % save_interval == 0 and step > 0:
            print(f"Progress: {step}/{total_timesteps} steps completed")

    # 关闭环境
    env.close()


# 调试函数：详细分析观察空间组成
def debug_observation_space(model_path: str):
    """
    调试函数：详细分析观察空间的组成:cite[5]
    """
    env = UnitreeGo2Env(model_path)
    observation = env.reset()

    print("=" * 50)
    print("OBSERVATION SPACE DEBUG INFO")
    print("=" * 50)

    # 输出各组件维度
    print(f"Total observation dimension: {len(observation)}")
    print(f"qpos dimension: {env.model.nq}")
    print(f"qvel dimension: {env.model.nv}")
    print(f"ctrl dimension: {env.model.nu}")
    print(f"xquat dimension: {env.data.xquat[1].shape}")
    print(f"xpos dimension: {env.data.xpos[1].shape}")

    # 输出各组件的具体值
    print("\nObservation components:")
    print(f"qpos: {env.data.qpos.flatten()}")
    print(f"qvel: {env.data.qvel.flatten()}")
    print(f"ctrl: {env.data.ctrl.flatten()}")
    print(f"body quat: {env.data.xquat[1]}")
    print(f"body pos: {env.data.xpos[1]}")

    # 验证重置后的观察值维度
    test_obs = env.reset()
    print(f"\nReset observation dimension: {len(test_obs)}")
    print(f"Consistent: {len(test_obs) == len(observation)}")

    env.close()


if __name__ == "__main__":
    # 模型路径 - 替换为你的Go2模型实际路径
    go2_model_path = "unitree_go2/scene.xml"

    # 调试观察空间
    debug_observation_space(go2_model_path)

    # 训练机器狗（取消注释以运行）
    # train_go2(go2_model_path, total_timesteps=1000000)