import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# 1. 创建网格
x, y = np.mgrid[-3:3:.05, -3:3:.05]  # 生成坐标点网格
pos = np.dstack((x, y))  # 转换为适合输入的形状

# 2. 定义不同的高斯分布参数（模拟B0, B1, B2, B3不同的场）
# 每个场的均值、协方差矩阵都不同，代表不同的“影响源”

# 场B0: 个体自身产生的场
mu_0 = [0.5, 0.5]
sigma_0 = [[1.0, 0.3], [0.3, 0.8]]
rv_0 = multivariate_normal(mu_0, sigma_0)

# 场B1: 人与人影响产生的场
mu_1 = [-1, -1]
sigma_1 = [[1.2, -0.5], [-0.5, 1.0]]
rv_1 = multivariate_normal(mu_1, sigma_1)

# 场B2: 社会影响产生的场 (可以更“宽”更“平”)
mu_2 = [0, 0]
sigma_2 = [[2.5, 0.0], [0.0, 2.5]]
rv_2 = multivariate_normal(mu_2, sigma_2)

# 3. 计算概率密度函数 (PDF)
z_0 = rv_0.pdf(pos)
z_1 = rv_1.pdf(pos)
z_2 = rv_2.pdf(pos)

# 4. 绘制单个场（例如B0）
plt.figure(figsize=(8, 6))
plt.contourf(x, y, z_0, levels=20, cmap='viridis') # 使用等高线填充图
plt.colorbar(label='Probability Density')
plt.title('Representation of Personal Field $B_0$ \n (2D Gaussian Distribution)')
plt.xlabel('X-axis (e.g., Spatial or Abstract Dimension)')
plt.ylabel('Y-axis (e.g., Spatial or Abstract Dimension)')
plt.grid(True, alpha=0.3)
plt.show()

# 5. 绘制合成场 (B0 + B1 + B2) - 近似总人势场
z_total = z_0 + z_1 + z_2 # 简单叠加来模拟场的合成

plt.figure(figsize=(8, 6))
contour = plt.contourf(x, y, z_total, levels=20, cmap='plasma')
plt.colorbar(contour, label='Combined Field Strength')
plt.title('Total Human Potential Field\n(Superposition of $B_0$, $B_1$, $B_2$)')
plt.xlabel('Dimension X')
plt.ylabel('Dimension Y')
# 在图上标记场中心
plt.scatter(mu_0[0], mu_0[1], c='red', s=100, marker='o', label='Source $B_0$ (Self)')
plt.scatter(mu_1[0], mu_1[1], c='blue', s=100, marker='s', label='Source $B_1$ (Others)')
plt.scatter(mu_2[0], mu_2[1], c='green', s=100, marker='^', label='Source $B_2$ (Society)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


# 继续使用上面的合成场 z_total

# 1. 在定义域内进行均匀随机采样（模拟古老的随机方法，如蓍草）
num_samples = 500  # 采样点数
sample_x = np.random.uniform(-3, 3, num_samples)
sample_y = np.random.uniform(-3, 3, num_samples)

# 2. 绘制采样点叠加在场上的效果
plt.figure(figsize=(10, 8))
# 绘制背景场
contour = plt.contourf(x, y, z_total, levels=20, alpha=0.6, cmap='Greys')
plt.colorbar(contour, label='Field Strength')
# 绘制随机采样点
plt.scatter(sample_x, sample_y, s=10, c='darkblue', alpha=0.7, label='Random Samples (Divination Process)')
# 标记场中心
plt.scatter(mu_0[0], mu_0[1], c='red', s=150, marker='*', edgecolors='white', label='Field Centers')
plt.scatter(mu_1[0], mu_1[1], c='red', s=150, marker='*', edgecolors='white')
plt.scatter(mu_2[0], mu_2[1], c='red', s=150, marker='*', edgecolors='white')

plt.title('Monte Carlo Sampling of the Human Potential Field\n(A Mathematical Analogy to Divination)')
plt.xlabel('Dimension X')
plt.ylabel('Dimension Y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 3. （高级）可以计算采样点处场的均值，作为对场的估计
# 这里只是演示，更复杂的估计需要更精细的算法
sample_values = []
for i in range(num_samples):
    # 查找每个采样点最近的网格点对应的场强度（简化处理）
    idx_x = np.argmin(np.abs(x[0, :] - sample_x[i]))
    idx_y = np.argmin(np.abs(y[:, 0] - sample_y[i]))
    sample_values.append(z_total[idx_y, idx_x]) # 注意索引顺序

estimated_field_strength = np.mean(sample_values)
print(f"Estimated average field strength from {num_samples} samples: {estimated_field_strength:.4f}")


# 这是一个更示意性的图，表示输入（采样）通过模型（易经）得到输出（卦象/决策）

plt.figure(figsize=(12, 6))

# 定义节点位置
input_pos = (0.1, 0.5)
model_pos = (0.5, 0.5)
output_pos = (0.9, 0.5)

# 绘制节点
plt.scatter(*input_pos, s=1000, c='lightblue', edgecolors='darkblue', zorder=5)
plt.scatter(*model_pos, s=2000, c='lightcoral', edgecolors='darkred', zorder=5)
plt.scatter(*output_pos, s=1000, c='lightgreen', edgecolors='darkgreen', zorder=5)

# 在节点下方添加中文文字标签
plt.text(input_pos[0], input_pos[1]-0.15, '输入\n(占卜随机采样)',
         ha='center', va='top', fontweight='bold', fontsize=10,
         fontfamily='SimHei')  # 使用黑体确保中文显示

plt.text(model_pos[0], model_pos[1]-0.2, '模型 $M_0$\n(易经智慧)',
         ha='center', va='top', fontweight='bold', fontsize=10,
         fontfamily='SimHei')

plt.text(output_pos[0], output_pos[1]-0.15, '输出\n(卦象指引)',
         ha='center', va='top', fontweight='bold', fontsize=10,
         fontfamily='SimHei')

# 绘制箭头
plt.arrow(input_pos[0]+0.05, input_pos[1], model_pos[0]-input_pos[0]-0.1, 0,
          head_width=0.03, head_length=0.02, fc='k', ec='k', length_includes_head=True)
plt.arrow(model_pos[0]+0.05, model_pos[1], output_pos[0]-model_pos[0]-0.1, 0,
          head_width=0.03, head_length=0.02, fc='k', ec='k', length_includes_head=True)

# 美化
plt.title('占卜作为模型推理过程', fontsize=14, fontfamily='SimHei')
plt.xlim(0, 1)
plt.ylim(0.2, 0.8)  # 调整Y轴范围，为下方的文字留出空间
plt.axis('off')  # 关闭坐标轴
plt.tight_layout()
plt.show()