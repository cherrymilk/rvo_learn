import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import rvo23d


# Agent类
class Agent:
    def __init__(self, position, radius=0.1, color_str="red"):
        self.position = np.array(position, dtype=np.float64)  # 位置
        self.radius = radius                                  # 半径
        self.velocity = np.zeros(3)                           # 当前速度指令，初始化为0
        self.color_str = color_str
        self.target = np.zeros(3)
        self.history_points = []

    def update(self, dt=0.1):
        # 简单的位置更新
        self.position += self.velocity * dt
        self.history_points.append(self.position.copy())

    def update_vel(self, velocity: np.ndarray):
        self.velocity = velocity

    def update_target(self, target: np.ndarray):
        self.target = target

    def prefer_vel(self):
        vel_ = self.target - self.position
        return vel_[0], vel_[1], vel_[2]

    def get_hist_traj(self):
        hist_traj = np.vstack(self.history_points)
        return hist_traj


class Obstacle:
    def __init__(self, pts: np.ndarray, color_str="lightblue"):
        self.pts = pts
        self.color_str = color_str


# 生成多个Agent
num_agents = 5
circle_radius = 1.8
time_length = 5
fps = 30
total_steps = time_length*fps
time_step = 1/fps
max_vel = 2
circle_colors = [
    'red', 'blue', 'green', 'purple', 'orange',
    'brown', 'pink', 'gray', 'olive', 'cyan'
]
agents = []
rvo_agent_idx = []
# Create simulator
sim = rvo23d.PyRVOSimulator(time_step, 1.5, 5, 2, 0.4, max_vel)
for i in range(num_agents):
    # 计算每个机器人在圆周上的角度（弧度制）
    angle = 2 * np.pi * i / num_agents

    # 计算机器人的x和y坐标
    x = 0 + circle_radius * np.cos(angle)
    y = 0 + circle_radius * np.sin(angle)

    each_agent = Agent(position=np.array([x, y, 0]), radius=50.0, color_str=circle_colors[i])
    each_agent.update_target(target=np.array([-x, -y, 0]))
    agents.append(each_agent)
    rvo_agent_idx.append(sim.addAgent((x, y, 0)))

# Time forward
for i in range(total_steps):
    for j in range(len(agents)):
        # Set agent position
        sim.setAgentPosition(rvo_agent_idx[j], (agents[j].position[0], agents[j].position[1], agents[j].position[2]))
        # Set preferred velocity
        sim.setAgentPrefVelocity(rvo_agent_idx[j], agents[j].prefer_vel())
    # ORCA calculation
    sim.doStep()
    # Fetch velocity commands from the ORCA
    for j in range(len(agents)):
        agent = agents[j]
        agent.update_vel(np.asarray(sim.getAgentVelocity(rvo_agent_idx[j])))
        agent.update(dt=time_step)

    # 创建图形和坐标轴
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 使用for循环绘制圆圈
    for j in range(len(agents)):
        ax.scatter(
            agents[j].get_hist_traj()[:, 0],
            agents[j].get_hist_traj()[:, 1],
            agents[j].get_hist_traj()[:, 2],
            s=agents[j].radius,
            c=agents[j].color_str,
            edgecolors='black',
            linewidths=1.5,
            alpha=0.7,
            label=f'Circle_{j+1}'
        )
    # 使用for循环绘制叉号
    for j in range(len(agents)):
        ax.scatter(
            agents[j].target[0],
            agents[j].target[1],
            agents[j].target[2],
            s=agents[j].radius,
            c=agents[j].color_str,
            marker='x',
            linewidths=2,
            alpha=0.9,
            label=f'Target_{j+1}'
        )

    # 添加标题和标签
    ax.set_title('Time = {} s'.format(round((i+1)/fps, 2)), fontsize=16)
    ax.set_xlabel('X', fontsize=14)
    ax.set_ylabel('Y', fontsize=14)
    ax.set_zlabel('Z', fontsize=14)
    ax.set_box_aspect([1, 1, 1])  # 保持XYZ比例一致
    # 设置坐标轴范围
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    # 显示图形
    plt.tight_layout()
    plt.savefig("images_3d/{}.png".format(i))
    plt.close()

