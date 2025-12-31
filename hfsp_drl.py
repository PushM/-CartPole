import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import glob
import matplotlib.pyplot as plt
from collections import deque

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ==========================================
# 1. 混合流水车间调度环境 (HFSP Environment)
# ==========================================
class HFSPEnv(gym.Env):
    def __init__(self, num_jobs=20, num_stages=3, machines_per_stage=[2, 2, 2], job_times=None):
        super(HFSPEnv, self).__init__()
        self.num_jobs = num_jobs
        self.num_stages = num_stages
        self.machines_per_stage = machines_per_stage
        self.initial_job_times = job_times
        if self.initial_job_times is not None:
            self.job_times = self.initial_job_times.copy()
        
        # 动作空间：30 个复合调度规则 (6 工件规则 * 5 机器规则)
        self.action_space = gym.spaces.Discrete(30)
        
        # 状态空间：
        # 按照论文 3.3.2 节设计 8 类特征
        # f1: 队列长度比 (1)
        # f2: 阶段结束时间比 (1)
        # f3: 累积平均负载比 (1)
        # f4: 队列最大加工时间比 (1)
        # f5: 队列最小加工时间比 (1)
        # f6: 最小工时比 (当前/下一阶段) (1)
        # f7: 最大工时比 (当前/下一阶段) (1)
        # f8: 机器结束时间比 (统计值: Mean, Std, Min, Max) (4)
        # 加上 阶段 One-Hot (num_stages)
        self.state_dim = self.num_stages + 11
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.next_stage_idx = 0

        # self.next_machine_idx = 0  <-- 机器现在由 Action 动态选择，不再由 Environment 预选


    def reset(self):
        # 生成作业数据
        if self.initial_job_times is not None:
            self.job_times = self.initial_job_times.copy()
        else:
            self.job_times = np.random.randint(1, 10, size=(self.num_jobs, self.num_stages))
        
        self.machine_available_time = [[0] * m for m in self.machines_per_stage]
        self.queues = [[] for _ in range(self.num_stages)]
        self.queues[0] = list(range(self.num_jobs))
        self.job_completion_times = np.zeros((self.num_jobs, self.num_stages))
        
        # 预计算全局统计量 (用于状态归一化)
        # 1. 每个阶段的平均加工时间 p_bar_j
        self.avg_proc_times = np.mean(self.job_times, axis=0)
        
        # 2. 全局工时比最大值 (用于 f6, f7 归一化)
        # ratio_i_j = p_{i,j} / p_{i,j+1}
        self.max_proc_ratios = np.ones(self.num_stages) # default 1
        for s in range(self.num_stages - 1):
            # 避免除以 0
            next_times = self.job_times[:, s+1].copy()
            next_times[next_times == 0] = 1e-5
            ratios = self.job_times[:, s] / next_times
            self.max_proc_ratios[s] = np.max(ratios) if len(ratios) > 0 else 1.0
        
        # 3. 奖励归一化因子 (最大加工时间作为参考，用于论文公式 3.19)
        self.reward_scale = np.max(self.job_times) if self.job_times.size > 0 else 1.0

        self.current_time = 0
        self.completed_jobs = 0
        self.previous_makespan = 0
        
        # 推进到第一个决策点
        self._advance_to_decision()
        
        return self._get_state()

    def _advance_to_decision(self):
        # 推进时间，直到有机器空闲且队列不为空
        while self.completed_jobs < self.num_jobs:
            # 1. 检查任意阶段是否有待处理工件且有机器空闲
            # 这里的逻辑需要微调：以前是选定一个机器，现在只需要选定一个阶段
            # 只要某阶段队列不为空，且存在至少一台机器空闲，就可以进行决策
            candidates = []
            for s in range(self.num_stages):
                if len(self.queues[s]) > 0:
                    min_time = min(self.machine_available_time[s])
                    if min_time <= self.current_time:
                        candidates.append(s)
            
            if candidates:
                # 简单策略：优先调度靠后的阶段 (First Valid)
                self.next_stage_idx = candidates[0]
                return

            # 2. 如果没有，推进时间到下一个机器释放点
            future_times = [t for stage_times in self.machine_available_time for t in stage_times if t > self.current_time]
            if not future_times:
                pass # Should not happen usually
            else:
                self.current_time = min(future_times)

    def _get_state(self):
        # 实现论文 3.3.2 节定义的 8 个状态特征
        s_idx = self.next_stage_idx
        queue = self.queues[s_idx]
        
        # 1. Stage One-Hot
        stage_vec = np.zeros(self.num_stages, dtype=np.float32)
        stage_vec[s_idx] = 1.0
        
        # 如果所有工件都做完了，返回全0 (或保持最后状态)
        if self.completed_jobs == self.num_jobs:
            return np.zeros(self.state_dim, dtype=np.float32)

        # 辅助计算：当前阶段所有机器的结束时间
        # machine_available_time[s] 存储的是机器可用时间（即上一任务结束时间）
        stage_machine_times = self.machine_available_time[s_idx]
        max_stage_time = max(stage_machine_times) if stage_machine_times else 1e-5
        if max_stage_time == 0: max_stage_time = 1e-5
        
        # 全局最大结束时间 (所有阶段)
        global_max_time = max([max(m_times) for m_times in self.machine_available_time])
        if global_max_time == 0: global_max_time = 1e-5

        # --- 特征计算 ---
        
        # f1: 队列工件数量比例
        f1 = len(queue) / self.num_jobs
        
        # f2: 当前阶段最大结束时间 / 全局最大结束时间
        f2 = max_stage_time / global_max_time
        
        # f3: 累积平均负载比
        # 公式 (3.13): (Sum_{k=1}^j Sum_{i in Q_k} p_{i,j}) / (p_bar_j * Sum_{k=1}^j |Q_k|)
        # 解释：考察当前阶段以及之前所有阶段的队列。
        # 分子：这些队列中的工件，在*当前阶段 j* 的加工时间总和。
        # 分母：当前阶段 j 的平均加工时间 * 这些队列的总人数。
        numer_f3 = 0.0
        denom_count = 0
        for k in range(s_idx + 1): # 0 to s_idx
            q_k = self.queues[k]
            denom_count += len(q_k)
            for job_id in q_k:
                numer_f3 += self.job_times[job_id, s_idx]
        
        avg_pj = self.avg_proc_times[s_idx]
        if avg_pj == 0: avg_pj = 1e-5
        if denom_count == 0:
            f3 = 0.0
        else:
            f3 = numer_f3 / (avg_pj * denom_count)

        # 针对当前队列的统计 (如果队列为空，给 0)
        if len(queue) > 0:
            q_proc_times = [self.job_times[j, s_idx] for j in queue]
            max_p = np.max(q_proc_times)
            min_p = np.min(q_proc_times)
            
            # f4: Max Proc / Avg Proc (Avg of current stage global)
            f4 = max_p / avg_pj
            
            # f5: Min Proc / Avg Proc
            f5 = min_p / avg_pj
            
            # f6 & f7: Proc(j) / Proc(j+1) ratios
            if s_idx < self.num_stages - 1:
                ratios = []
                for j in queue:
                    p_curr = self.job_times[j, s_idx]
                    p_next = self.job_times[j, s_idx+1]
                    if p_next == 0: p_next = 1e-5
                    ratios.append(p_curr / p_next)
                
                min_r = np.min(ratios)
                max_r = np.max(ratios)
                global_max_r = self.max_proc_ratios[s_idx]
                if global_max_r == 0: global_max_r = 1e-5
                
                f6 = min_r / global_max_r
                f7 = max_r / global_max_r
            else:
                f6 = 0.0 # Last stage has no next stage
                f7 = 0.0
        else:
            f4, f5, f6, f7 = 0.0, 0.0, 0.0, 0.0

        # f8: 机器负载分布 (统计特征)
        # 原文是针对特定机器 l 的 f_{l,8} = EndTime_l / Max_Stage_EndTime
        # 为了作为状态输入，我们取这些比率的统计值: Mean, Std, Min, Max
        machine_ratios = [t / max_stage_time for t in stage_machine_times]
        f8_mean = np.mean(machine_ratios)
        f8_std = np.std(machine_ratios)
        f8_min = np.min(machine_ratios)
        f8_max = np.max(machine_ratios) # Should be 1.0 usually
        
        # 组合特征向量
        features = np.array([f1, f2, f3, f4, f5, f6, f7, f8_mean, f8_std, f8_min, f8_max], dtype=np.float32)
        
        return np.concatenate([stage_vec, features])

    def step(self, action):
        if self.completed_jobs == self.num_jobs:
            return self._get_state(), 0, True, {"makespan": self.previous_makespan}

        # 解析动作：Action (0-29) -> (JobRule, MachineRule)
        # Job Rules: 0-5
        # Machine Rules: 0-4
        job_rule_idx = action // 5
        machine_rule_idx = action % 5
        
        stage_idx = self.next_stage_idx
        queue = self.queues[stage_idx] # list of job_ids
        
        # --- 1. 执行工件选择规则 ---
        # 预计算所有候选工件的相关属性
        # job_times[j, s]: 当前阶段加工时间
        # sum(job_times[j, s+1:]): 后继加工时间
        # total_time: sum(job_times[j, :])
        
        candidates = []
        for j in queue:
            proc_time = self.job_times[j, stage_idx]
            rem_time = np.sum(self.job_times[j, stage_idx+1:]) if stage_idx < self.num_stages - 1 else 0
            total_time = np.sum(self.job_times[j, :])
            candidates.append({
                'id': j,
                'proc': proc_time,
                'rem': rem_time,
                'ratio': proc_time / total_time if total_time > 0 else 0
            })
            
        selected_job_id = -1
        
        if job_rule_idx == 0: # SPT: 最短加工时间
            selected_job_id = min(candidates, key=lambda x: x['proc'])['id']
        elif job_rule_idx == 1: # LPT: 最长加工时间
            selected_job_id = max(candidates, key=lambda x: x['proc'])['id']
        elif job_rule_idx == 2: # SRPT: 最短后继时间
            selected_job_id = min(candidates, key=lambda x: x['rem'])['id']
        elif job_rule_idx == 3: # LRPT: 最长后继时间
            selected_job_id = max(candidates, key=lambda x: x['rem'])['id']
        elif job_rule_idx == 4: # 最小比率 (Proc / Total)
            selected_job_id = min(candidates, key=lambda x: x['ratio'])['id']
        elif job_rule_idx == 5: # 最大比率 (Proc / Total)
            selected_job_id = max(candidates, key=lambda x: x['ratio'])['id']
        else:
            selected_job_id = candidates[0]['id'] # Fallback
            
        self.queues[stage_idx].remove(selected_job_id)
        
        # --- 2. 执行机器选择规则 ---
        # 候选机器：所有机器 (包括非空闲的，因为可以插入队列，或者等待释放)
        # 但通常调度是针对"何时开始"，这里简化为：只能选当前阶段的机器
        # 既然 _advance_to_decision 保证了至少有一台机器空闲 (或者即将空闲)，
        # 我们需要计算每台机器如果处理该工件，会发生什么。
        
        machines = []
        num_machines = self.machines_per_stage[stage_idx]
        proc_time = self.job_times[selected_job_id, stage_idx]
        
        # 工件到达时间 (上一阶段完成时间)
        arrival_time = 0
        if stage_idx > 0:
            arrival_time = self.job_completion_times[selected_job_id, stage_idx - 1]
            
        for m in range(num_machines):
            avail_time = self.machine_available_time[stage_idx][m]
            start_time = max(self.current_time, avail_time, arrival_time)
            finish_time = start_time + proc_time
            idle_added = max(0, start_time - avail_time) # 空闲时间增加量
            
            machines.append({
                'idx': m,
                'avail': avail_time,
                'finish': finish_time,
                'idle': idle_added
            })
            
        selected_machine_idx = -1
        
        if machine_rule_idx == 0: # 最大结束时间 (为什么选最大的？可能为了负载均衡？按论文实现)
            selected_machine_idx = max(machines, key=lambda x: x['avail'])['idx'] 
            # 注意：原文说“加工结束时间最大的机器”，可能是指 avail_time 最大，也可能是 finish_time 最大。
            # 通常为了最小化 Cmax，应该选 finish_time 最小的。
            # 但既然原文写了“最大”，可能是为了填补空隙？或者原文是反向规则？
            # 我们这里严格按字面意思：Avail Time Max (当前最忙的机器？)
            # 或者是指 Finish Time Max。我们假设是 Avail Time。
            pass
        
        # 修正理解：通常规则是为了优化。
        # Rule 1: Max Completion Time (of the machine currently). -> 可能是为了利用碎片？
        # Rule 2: Min Completion Time. -> 标准 Greedy
        # Rule 3: Min Idle Time Increment.
        # Rule 4: Min (Idle + Proc).
        # Rule 5: Min Max Completion Time (Makespan increment).
        
        if machine_rule_idx == 0: # 1. 当前阶段加工结束时间最大 (Avail Time Max)
            selected_machine_idx = max(machines, key=lambda x: x['avail'])['idx']
        elif machine_rule_idx == 1: # 2. 当前阶段加工结束时间最小 (Avail Time Min) - 标准贪心
            selected_machine_idx = min(machines, key=lambda x: x['avail'])['idx']
        elif machine_rule_idx == 2: # 3. 空闲时间增加最少
            selected_machine_idx = min(machines, key=lambda x: x['idle'])['idx']
        elif machine_rule_idx == 3: # 4. (空闲 + 加工) 最少 -> 其实就是 Finish - Avail 最少？
            # Idle + Proc = (Start - Avail) + Proc = Finish - Avail.
            selected_machine_idx = min(machines, key=lambda x: x['idle'] + proc_time)['idx']
        elif machine_rule_idx == 4: # 5. 当前阶段最大结束时间最短 (Min Finish Time)
            selected_machine_idx = min(machines, key=lambda x: x['finish'])['idx']
        else:
            selected_machine_idx = 0
            
        # --- 3. 更新状态 ---
        m_idx = selected_machine_idx
        
        # 重新计算最终的时间 (因为上面是模拟)
        real_avail = self.machine_available_time[stage_idx][m_idx]
        real_start = max(self.current_time, real_avail, arrival_time)
        real_finish = real_start + proc_time
        
        self.machine_available_time[stage_idx][m_idx] = real_finish
        self.job_completion_times[selected_job_id, stage_idx] = real_finish
        
        if stage_idx < self.num_stages - 1:
            self.queues[stage_idx + 1].append(selected_job_id)
        else:
            self.completed_jobs += 1
            
        # 4. 计算奖励 (基于论文 3.3.3 节: 空闲时间增量)
        # 获取被选机器产生的空闲时间
        # machines 列表按索引顺序对应 machine 0..M-1
        selected_machine_entry = machines[selected_machine_idx]
        idle_added = selected_machine_entry['idle']
        
        # 公式 (3.18): r = idletime(t) - idletime(t+1)
        # idletime 增加，则 r 为负。r = - (NewIdle - OldIdle) = - idle_added
        raw_reward = -idle_added
        
        # 公式 (3.19): 归一化到 [-1, 1]
        # 使用全局最大加工时间作为归一化分母的估计
        normalized_reward = raw_reward / self.reward_scale
        
        # 确保在 [-1, 1] 范围内 (通常 raw_reward <= 0)
        reward = np.clip(normalized_reward, -1.0, 1.0)
        
        # 更新 previous_makespan 用于 logging
        current_makespan = max([max(stage) for stage in self.machine_available_time])
        self.previous_makespan = current_makespan 
        
        done = (self.completed_jobs == self.num_jobs)
        
        if not done:
            self._advance_to_decision()
            
        return self._get_state(), reward, done, {"makespan": current_makespan}

# ==========================================
# 2. Neural Network & Agents
# ==========================================
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.capacity = capacity

    def add(self, error, sample):
        p = (error + 1e-5) ** self.alpha
        self.tree.add(p, sample)

    def sample(self, n, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight

    def update(self, idx, error):
        p = (error + 1e-5) ** self.alpha
        self.tree.update(idx, p)
    
    def __len__(self):
        return self.tree.n_entries

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Value stream
        self.fc_value = nn.Linear(128, 1)
        
        # Advantage stream
        self.fc_advantage = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        val = self.fc_value(x)
        adv = self.fc_advantage(x)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        return val + adv - adv.mean(dim=1, keepdim=True)

class DDTD_Agent:
    def __init__(self, state_dim, action_dim, max_episodes=800):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = 1e-3
        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilon_min = 0.1
        self.epsilon_decay_step = (self.epsilon - self.epsilon_min) / max_episodes
        self.batch_size = 100
        # Use Prioritized Replay Buffer
        self.memory = PrioritizedReplayBuffer(capacity=3000)
        self.beta = 0.4
        self.beta_increment = (1.0 - 0.4) / max_episodes
        self.tau = 0.1
        
        # Use Dueling Network
        self.q_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss(reduction='none') # Need element-wise loss for PER

    def select_action(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return torch.argmax(q_values).item()

    def store_transition(self, s, a, r, ns, d):
        # Initial priority: max priority or high constant
        error = 1.0 # High priority for new experiences
        self.memory.add(error, (s, a, r, ns, d))

    def update(self):
        if len(self.memory) < self.batch_size: return
        
        # Sample from PER
        batch, idxs, is_weights = self.memory.sample(self.batch_size, self.beta)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1)
        
        # Double DQN Logic
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values
            
        current_q = self.q_net(states).gather(1, actions)
        
        # Weighted Loss
        loss = (is_weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
        
        # Update Priorities
        errors = torch.abs(current_q - target_q).detach().numpy()
        for i in range(self.batch_size):
            self.memory.update(idxs[i], errors[i][0])
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_step
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        self.soft_update_target_network()

    def soft_update_target_network(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, filename):
        torch.save(self.q_net.state_dict(), filename)

class DQN_Agent(DDTD_Agent):
    def update(self):
        if len(self.memory) < self.batch_size: return
        
        batch, idxs, is_weights = self.memory.sample(self.batch_size, self.beta)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        is_weights = torch.FloatTensor(is_weights).unsqueeze(1)

        # Standard DQN Logic: max_a Q(s', a)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        current_q = self.q_net(states).gather(1, actions)
        
        loss = (is_weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
        
        errors = torch.abs(current_q - target_q).detach().numpy()
        for i in range(self.batch_size):
            self.memory.update(idxs[i], errors[i][0])
            
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_step
        self.beta = min(1.0, self.beta + self.beta_increment)
        self.soft_update_target_network()

class DeepSarsa_Agent:
    def __init__(self, state_dim, action_dim, max_episodes=800):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = 1e-3
        self.gamma = 0.9
        self.epsilon = 0.9
        self.epsilon_min = 0.1
        self.epsilon_decay_step = (self.epsilon - self.epsilon_min) / max_episodes
        self.batch_size = 100
        self.memory = deque(maxlen=3000)
        self.tau = 0.1 
        
        self.q_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return torch.argmax(q_values).item()

    def store_transition(self, s, a, r, ns, na, d):
        self.memory.append((s, a, r, ns, na, d))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, next_actions, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        next_actions = torch.LongTensor(next_actions).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        current_q = self.q_net(states).gather(1, actions)
        
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_step
            
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, filename):
        torch.save(self.q_net.state_dict(), filename)

# ==========================================
# 2.6 NEH Heuristic
# ==========================================
def neh_algorithm(num_jobs, num_stages, job_times):
    # 1. 计算每个工件的总加工时间
    total_times = np.sum(job_times, axis=1)
    
    # 2. 按总加工时间降序排序
    sorted_jobs = np.argsort(total_times)[::-1]
    
    # 3. 逐个插入构建序列
    current_seq = []
    
    for job in sorted_jobs:
        best_makespan = float('inf')
        best_pos = -1
        
        # 尝试插入到 current_seq 的每一个可能位置
        for pos in range(len(current_seq) + 1):
            temp_seq = current_seq[:pos] + [job] + current_seq[pos:]
            
            # 快速计算 Makespan (简化版模拟)
            ms = calculate_makespan_from_sequence(temp_seq, num_stages, job_times)
            
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, job)
        
    return best_makespan

def calculate_makespan_from_sequence(seq, num_stages, job_times, machines_per_stage):
    # Simulate the schedule to calculate makespan
    # seq: list of job IDs (order for Stage 0)
    
    current_jobs = list(seq)
    # Track completion time of each job at the previous stage (initially 0)
    job_completion = {j: 0 for j in seq}
    
    # Track available time of each machine at each stage
    machine_avail = [[0] * m for m in machines_per_stage]
    
    for s in range(num_stages):
        if s > 0:
            # For subsequent stages, process jobs in order of their arrival (completion at previous stage)
            # Stable sort preserves relative order for simultaneous arrivals (FIFO)
            current_jobs.sort(key=lambda j: job_completion[j])
        
        for job_id in current_jobs:
            # Find earliest available machine in current stage
            stage_machines = machine_avail[s]
            best_m = 0
            min_t = stage_machines[0]
            for m in range(1, len(stage_machines)):
                if stage_machines[m] < min_t:
                    min_t = stage_machines[m]
                    best_m = m
            
            arrival = job_completion[job_id]
            start = max(min_t, arrival)
            finish = start + job_times[job_id, s]
            
            machine_avail[s][best_m] = finish
            job_completion[job_id] = finish
            
    # Makespan is the max availability time at the last stage
    return max(machine_avail[-1])

# ==========================================
# 3. 训练与对比
# ==========================================
def load_fernandez_instance(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        all_numbers = []
        for line in lines:
            all_numbers.extend([int(x) for x in line.strip().split()])
            
        if not all_numbers: return None, None, None, None
        
        iterator = iter(all_numbers)
        try:
            num_jobs = next(iterator)
            num_stages = next(iterator)
            
            machines_per_stage = []
            for _ in range(num_stages):
                machines_per_stage.append(next(iterator))
                
            job_times = np.zeros((num_jobs, num_stages))
            # Data is Stage-Major: Stage 1 (all jobs), Stage 2 (all jobs)...
            for s in range(num_stages):
                for j in range(num_jobs):
                    job_times[j, s] = next(iterator)
        except StopIteration:
            print(f"Error parsing {file_path}: unexpected end of file")
            return None, None, None, None
            
        return num_jobs, num_stages, machines_per_stage, job_times
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None, None

def evaluate_sequence(env, sequence):
    return calculate_makespan_from_sequence(sequence, env.num_stages, env.job_times, env.machines_per_stage)

def run_neh(env):
    total_times = np.sum(env.job_times, axis=1)
    sorted_jobs = np.argsort(total_times)[::-1]
    current_seq = []
    for job in sorted_jobs:
        best_ms = float('inf')
        best_seq = []
        for pos in range(len(current_seq) + 1):
            temp_seq = current_seq[:pos] + [job] + current_seq[pos:]
            ms = evaluate_sequence(env, temp_seq)
            if ms < best_ms:
                best_ms = ms
                best_seq = temp_seq
        current_seq = best_seq
    return best_ms

def run_single_instance(file_path):
    filename = os.path.basename(file_path)
    print(f"Processing: {filename}")

    num_jobs, num_stages, machines_per_stage, job_times = load_fernandez_instance(file_path)
    if num_jobs is None: return None

    env = HFSPEnv(num_jobs=num_jobs, num_stages=num_stages, machines_per_stage=machines_per_stage, job_times=job_times)
    num_episodes = 800
    
    # Create models directory
    if not os.path.exists("trained_models"):
        os.makedirs("trained_models")

    # 1. NEH
    neh_cmax = run_neh(env)

    # 2. Standard DQN
    agent_dqn = DQN_Agent(env.state_dim, env.action_space.n, max_episodes=num_episodes)
    best_dqn = float('inf')
    global_step = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        ms = 0
        while not done:
            global_step += 1
            action = agent_dqn.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent_dqn.store_transition(state, action, reward, next_state, done)
            if global_step % 10 == 0:
                agent_dqn.update()
            state = next_state
            ms = info['makespan']
        if ms < best_dqn: best_dqn = ms
    agent_dqn.save_model(f"trained_models/{filename}_DQN.pth")

    # 3. Deep Sarsa (DS)
    agent_ds = DeepSarsa_Agent(env.state_dim, env.action_space.n, max_episodes=num_episodes)
    best_ds = float('inf')
    global_step = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        ms = 0
        action = agent_ds.select_action(state)
        while not done:
            global_step += 1
            next_state, reward, done, info = env.step(action)
            ms = info['makespan']
            if not done:
                next_action = agent_ds.select_action(next_state)
                agent_ds.store_transition(state, action, reward, next_state, next_action, done)
                if global_step % 10 == 0:
                    agent_ds.update()
                state = next_state
                action = next_action
            else:
                agent_ds.store_transition(state, action, reward, next_state, 0, done)
                if global_step % 10 == 0:
                    agent_ds.update()
        if ms < best_ds: best_ds = ms
    agent_ds.save_model(f"trained_models/{filename}_DS.pth")

    # 4. DDTD (Ours)
    agent_ddtd = DDTD_Agent(env.state_dim, env.action_space.n, max_episodes=num_episodes)
    best_ddtd = float('inf')
    global_step = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        ms = 0
        while not done:
            global_step += 1
            action = agent_ddtd.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent_ddtd.store_transition(state, action, reward, next_state, done)
            if global_step % 10 == 0:
                agent_ddtd.update()
            state = next_state
            ms = info['makespan']
        if ms < best_ddtd: best_ddtd = ms
    agent_ddtd.save_model(f"trained_models/{filename}_DDTD.pth")

    # RPD Calculation
    best_known = min(neh_cmax, best_dqn, best_ds, best_ddtd)
    
    rpd_neh = (neh_cmax - best_known) / best_known * 100 if best_known > 0 else 0
    rpd_dqn = (best_dqn - best_known) / best_known * 100 if best_known > 0 else 0
    rpd_ds = (best_ds - best_known) / best_known * 100 if best_known > 0 else 0
    rpd_ddtd = (best_ddtd - best_known) / best_known * 100 if best_known > 0 else 0

    try:
        parts = filename.replace('.txt', '').split('_')
        if len(parts) >= 3:
            short_name = f"{parts[-3]}_{parts[-2]}_{parts[-1]}"
            n, m, idx = int(parts[-3]), int(parts[-2]), int(parts[-1])
        else: 
            short_name = filename
            n, m, idx = 0, 0, 0
    except: 
        short_name = filename
        n, m, idx = 0, 0, 0

    print(f"Processed {short_name}: NEH={neh_cmax:.0f}, DQN={best_dqn:.0f}, DS={best_ds:.0f}, DDTD={best_ddtd:.0f}")

    return {
        "short_name": short_name,
        "n_jobs": n, "n_stages": m, "inst_id": idx,
        "NEH": neh_cmax, "RPD_NEH": rpd_neh,
        "DQN": best_dqn, "RPD_DQN": rpd_dqn,
        "DS": best_ds,   "RPD_DS": rpd_ds,
        "DDTD": best_ddtd, "RPD_DDTD": rpd_ddtd,
        "ddtd_best": best_ddtd,
        "random_best": 0, # Not used in table 3.2 logic but kept for struct
        "rpd_ddtd": rpd_ddtd
    }

def run_comparison():
    data_dir = "Small_Size_Instances"
    if not os.path.exists(data_dir): data_dir = r"d:\project\CartPole\Small_Size_Instances"
    if not os.path.exists(data_dir): return
    
    instance_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    # target_instances = {
    #     "10_5_4", "10_10_5", "10_15_1", "10_20_8",
    #     "15_5_9", "15_10_2", "15_15_7", "15_20_6",
    #     "20_5_4", "20_10_5", "20_15_1", "20_20_8",
    #     "25_5_9", "25_10_2", "25_15_7", "25_20_6",
    #     "30_5_7", "30_10_1", "30_15_5", "30_20_7",
    #     "35_5_3", "35_10_4", "35_15_8", "35_20_6"
    # }
    target_instances = {
        "10_5_4", "10_10_5", "10_15_1", "10_20_8",
        # "15_5_9", "15_10_2", "15_15_7", "15_20_6",
        # "20_5_4", "20_10_5", "20_15_1", "20_20_8",
        # "25_5_9", "25_10_2", "25_15_7", "25_20_6",
        # "30_5_7", "30_10_1", "30_15_5", "30_20_7",
        # "35_5_3", "35_10_4", "35_15_8", "35_20_6"
    }
    
    selected_files = []
    for f in instance_files:
        try:
            core_name = os.path.basename(f).replace('instancia_', '').replace('.txt', '')
            if core_name in target_instances: selected_files.append(f)
        except: continue
            
    # Sort by Job, Stage, ID
    files_to_run = sorted(selected_files, key=lambda f: (
        int(os.path.basename(f).replace('instancia_', '').replace('.txt', '').split('_')[0]), 
        int(os.path.basename(f).replace('instancia_', '').replace('.txt', '').split('_')[1])
    ))
    
    print(f"Running comprehensive experiment on {len(files_to_run)} instances...")
    print("Methods: NEH, DQN, DeepSarsa (QS), DDTD (Ours)")
    
    summary = []
    for f in files_to_run:
        res = run_single_instance(f)
        if res: summary.append(res)
        
    # Output Table
    print("\n" + "="*100)
    print(f"{'Instance':<10} | {'Cmax':<30} || {'RPD':<30}")
    print(f"{'':<10} | {'NEH':<7} {'DQN':<7} {'QS':<7} {'DDTD':<7} || {'NEH':<7} {'DQN':<7} {'QS':<7} {'DDTD':<7}")
    print("-" * 100)
    
    for r in summary:
        print(f"{r['short_name']:<10} | {r['NEH']:<7.0f} {r['DQN']:<7.0f} {r['DS']:<7.0f} {r['DDTD']:<7.0f} || {r['RPD_NEH']:<7.2f} {r['RPD_DQN']:<7.2f} {r['RPD_DS']:<7.2f} {r['RPD_DDTD']:<7.2f}")
    print("="*100)
    
    # Also output in the other format if needed
    
if __name__ == '__main__':
    run_comparison()