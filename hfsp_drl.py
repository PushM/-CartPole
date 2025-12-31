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
        
        # 动作空间：选择队列中第 K 个工件 (按 SPT 排序)
        # 0: 选第1个 (SPT)
        # ...
        # 4: 选第5个
        self.k_choices = 5
        self.action_space = gym.spaces.Discrete(self.k_choices)
        
        # 状态空间：
        # 1. 阶段 One-Hot (num_stages)
        # 2. 全局/机器统计特征 (num_stages * 6 + 1)
        # 3. 候选工件特征 (Top-K jobs * 2 features: [ProcTime, RemTime])
        self.job_feat_dim = 2
        self.state_dim = num_stages + (num_stages * 6 + 1) + (self.k_choices * self.job_feat_dim)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

        self.next_stage_idx = 0
        self.next_machine_idx = 0

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
        
        self.current_time = 0
        self.completed_jobs = 0
        self.previous_makespan = 0
        
        # 推进到第一个决策点
        self._advance_to_decision()
        
        return self._get_state()

    def _advance_to_decision(self):
        # 推进时间，直到有机器空闲且队列不为空
        # 或者所有任务完成
        while self.completed_jobs < self.num_jobs:
            # 1. 检查当前时间点是否有可行调度
            candidates = []
            for s in range(self.num_stages):
                if len(self.queues[s]) > 0:
                    # 找该阶段最早空闲的机器
                    min_time = min(self.machine_available_time[s])
                    if min_time <= self.current_time:
                        m_idx = self.machine_available_time[s].index(min_time)
                        candidates.append((s, m_idx))
            
            if candidates:
                # 找到决策点！
                # 简单策略：优先调度靠后的阶段 (First Valid)
                # 也可以改为随机或固定顺序，这里取第一个
                self.next_stage_idx, self.next_machine_idx = candidates[0]
                return

            # 2. 如果没有，推进时间到下一个机器释放点
            future_times = [t for stage_times in self.machine_available_time for t in stage_times if t > self.current_time]
            if not future_times:
                # 理论上只有当所有任务都在队列中但无法处理（不可能）或已完成时才会到这
                # 但这里 completed_jobs < num_jobs，说明还有任务没完
                # 可能是任务还在前一阶段运行，还没到下一阶段队列
                # 这种情况下，我们找所有运行中任务的最早完成时间
                running_completion_times = []
                for j in range(self.num_jobs):
                    for s in range(self.num_stages):
                        # 如果任务 j 在 s 阶段完成时间 > current_time，说明它还在跑？
                        # 不完全是，job_completion_times 存的是历史。
                        # 我们需要推断系统的下一个事件时间。
                        # 简化：future_times 包含了所有机器释放时间。
                        # 如果 future_times 为空，说明所有机器都在 current_time 之前释放了，但队列为空？
                        # 这意味着所有剩余任务都在“传输中”？不，本模型没有传输时间。
                        # 意味着所有未完成任务都在前序阶段处理中。
                        pass
                break # Should not happen if logic is correct
            
            self.current_time = min(future_times)

    def _get_state(self):
        # 1. Stage One-Hot
        stage_vec = np.zeros(self.num_stages, dtype=np.float32)
        if self.completed_jobs < self.num_jobs:
            stage_vec[self.next_stage_idx] = 1.0
        
        # 2. 统计特征 (原有)
        stat_state = []
        for s in range(self.num_stages):
            valid_times = [t - self.current_time for t in self.machine_available_time[s] if t > self.current_time]
            if not valid_times: valid_times = [0.0]
            avg_load = np.mean(valid_times)
            std_load = np.std(valid_times) if len(valid_times) > 1 else 0.0
            queue_len = len(self.queues[s])
            stat_state.extend([avg_load, std_load, queue_len])
            
            if queue_len > 0:
                q_times = [self.job_times[j, s] for j in self.queues[s]]
                stat_state.extend([np.mean(q_times), np.max(q_times), np.min(q_times)])
            else:
                stat_state.extend([0.0, 0.0, 0.0])
        stat_state.append(self.completed_jobs / self.num_jobs)
        
        # 3. 候选工件特征 (Top-K)
        job_feats = []
        if self.completed_jobs < self.num_jobs:
            current_queue = self.queues[self.next_stage_idx]
            # 按 SPT (当前阶段加工时间) 排序
            # 我们需要获取工件ID
            sorted_indices = np.argsort([self.job_times[j, self.next_stage_idx] for j in current_queue])
            sorted_jobs = [current_queue[i] for i in sorted_indices]
            
            for k in range(self.k_choices):
                if k < len(sorted_jobs):
                    job_id = sorted_jobs[k]
                    proc_time = self.job_times[job_id, self.next_stage_idx]
                    rem_time = np.sum(self.job_times[job_id, self.next_stage_idx:])
                    job_feats.extend([proc_time, rem_time])
                else:
                    job_feats.extend([0.0, 0.0])
        else:
            job_feats = [0.0] * (self.k_choices * self.job_feat_dim)

        return np.concatenate([stage_vec, stat_state, job_feats])

    def step(self, action):
        if self.completed_jobs == self.num_jobs:
            return self._get_state(), 0, True, {"makespan": self.previous_makespan}

        # 1. 执行动作：从当前决策阶段的队列中选一个工件
        stage_idx = self.next_stage_idx
        machine_idx = self.next_machine_idx
        queue = self.queues[stage_idx]
        
        # 修正 Pop 逻辑
        # 我们重新获取排序后的 job_id
        # queue is a list of job_ids
        times = [(self.job_times[j, stage_idx], j) for j in queue]
        times.sort() # sort by time
        
        target_k = action if action < len(times) else len(times) - 1
        selected_job_id = times[target_k][1]
        
        self.queues[stage_idx].remove(selected_job_id)
        
        # 2. 计算时间
        proc_time = self.job_times[selected_job_id, stage_idx]
        start_time = max(self.current_time, self.machine_available_time[stage_idx][machine_idx])
        
        if stage_idx > 0:
            prev_finish = self.job_completion_times[selected_job_id, stage_idx - 1]
            start_time = max(start_time, prev_finish)
            
        finish_time = start_time + proc_time
        
        self.machine_available_time[stage_idx][machine_idx] = finish_time
        self.job_completion_times[selected_job_id, stage_idx] = finish_time
        
        if stage_idx < self.num_stages - 1:
            self.queues[stage_idx + 1].append(selected_job_id)
        else:
            self.completed_jobs += 1
            
        # 3. 计算奖励
        current_makespan = max([max(stage) for stage in self.machine_available_time])
        diff = current_makespan - self.previous_makespan
        self.previous_makespan = current_makespan
        # 放大奖励，避免过小
        reward = -diff if diff > 0 else 0.1 
        
        done = (self.completed_jobs == self.num_jobs)
        
        # 4. 推进到下一个决策点
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