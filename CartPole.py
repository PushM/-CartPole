import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# -----------------------------------
# 1. 定义模型 (演员和评论家)
# -----------------------------------
class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # 公共层：提取特征
        self.affine1 = nn.Linear(4, 128) # 输入状态维度为4
        
        # 演员层 (Actor): 输出动作的概率 (左或右)
        self.action_head = nn.Linear(128, 2)
        
        # 评论家层 (Critic): 输出当前状态的价值 (一个分数)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        
        # 演员计算动作概率 (softmax保证概率和为1)
        action_prob = F.softmax(self.action_head(x), dim=-1)
        
        # 评论家计算价值
        state_value = self.value_head(x)
        
        return action_prob, state_value

# -----------------------------------
# 2. 训练配置
# -----------------------------------
# 创建环境
env = gym.make('CartPole-v1')
model = ActorCritic()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = 1e-5 # 防止除以0的小数

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = model(state)
    
    # 演员根据概率随机抽样动作 (Exploration)
    m = Categorical(probs)
    action = m.sample()
    
    # 记录动作日志概率和评论家打分，用于后续更新
    model.saved_actions.append((m.log_prob(action), state_value))
    
    return action.item()

def finish_episode():
    """
    一局游戏结束后，进行学习 (更新参数)
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # 演员的损失
    value_losses = []  # 评论家的损失
    returns = []       # 实际获得的折算奖励

    # 计算真实的累积奖励 (从后往前推，因为越靠后的奖励受当前动作影响越小)
    # 也就是公式中的 Gt
    for r in model.rewards[::-1]:
        R = r + 0.99 * R # 0.99 是折扣因子 gamma
        returns.insert(0, R)
        
    returns = torch.tensor(returns)
    # 归一化奖励，让训练更稳定
    returns = (returns - returns.mean()) / (returns.std() + eps)

    # 计算损失
    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item() # 优势函数 = 实际奖励 - 评论家预估价值
        
        # 演员损失：如果优势大(R>value)，就增大该动作的概率(log_prob)
        # 负号是因为我们需要"最大化"奖励，而优化器是"最小化"损失
        policy_losses.append(-log_prob * advantage)
        
        # 评论家损失：让预估的价值(value)无限接近实际奖励(R)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # 梯度下降更新参数
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()

    # 清空历史数据
    del model.saved_actions[:]
    del model.rewards[:]

# -----------------------------------
# 3. 主循环
# -----------------------------------
def main():
    running_reward = 10
    
    # 训练直到智能体能坚持很久 (奖励达到阈值)
    for i_episode in range(1000):
        state, _ = env.reset()
        ep_reward = 0
        
        # 每一个时间步
        for t in range(1, 10000):
            action = select_action(state)
            state, reward, done, _, _ = env.step(action)
            
            model.rewards.append(reward)
            ep_reward += reward
            
            if done:
                break

        # 记录平滑后的奖励变化
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        
        # 更新模型
        finish_episode()
        
        if i_episode % 10 == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}')
        
        if running_reward > env.spec.reward_threshold:
            print(f"解决了! 在第 {i_episode} 局达到目标。")
            break

if __name__ == '__main__':
    main()