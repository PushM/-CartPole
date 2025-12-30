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
        
        # 动作空间：选择调度规则
        # 0: SPT (最短加工时间优先)
        # 1: LPT (最长加工时间优先)
        # 2: SRPT (剩余总加工时间最短优先)
        # 3: LRPT (剩余总加工时间最长优先)
        # 4: FIFO (先入先出)
        # 5: Random (随机)
        self.action_space = gym.spaces.Discrete(6)
        
        # 状态空间：特征向量
        # [每个阶段的平均机器负载, 每个阶段的排队任务数, 系统平均完工率]
        self.state_dim = num_stages * 2 + 1
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.state_dim,), dtype=np.float32)

    def reset(self):
        # 生成作业数据：[Job_ID, Stage_1_Time, Stage_2_Time, ...]
        if self.initial_job_times is not None:
            self.job_times = self.initial_job_times.copy()
        else:
            self.job_times = np.random.randint(1, 10, size=(self.num_jobs, self.num_stages))
        
        # 机器状态：记录每台机器的可用时间 [Stage][Machine_ID]
        self.machine_available_time = [[0] * m for m in self.machines_per_stage]
        
        # 队列状态：每个阶段的等待队列，存储 Job ID
        self.queues = [[] for _ in range(self.num_stages)]
        
        # 初始时，所有工件都在第0阶段的队列中
        self.queues[0] = list(range(self.num_jobs))
        
        # 记录每个工件在每个阶段的完成时间，用于计算 Makespan
        self.job_completion_times = np.zeros((self.num_jobs, self.num_stages))
        
        self.current_time = 0
        self.completed_jobs = 0
        
        return self._get_state()

    def _get_state(self):
        # 构建状态向量
        state = []
        
        # 1. 每个阶段的平均机器负载 (归一化)
        for s in range(self.num_stages):
            valid_times = [t - self.current_time for t in self.machine_available_time[s] if t > self.current_time]
            if valid_times:
                avg_load = np.mean(valid_times)
                state.append(avg_load)
            else:
                state.append(0.0)
            
        # 2. 每个阶段的排队数量
        for s in range(self.num_stages):
            state.append(len(self.queues[s]))
            
        # 3. 总体进度
        state.append(self.completed_jobs / self.num_jobs)
        
        return np.array(state, dtype=np.float32)

    def step(self, action):
        # 执行一次调度决策：
        # 找到第一个有空闲机器且队列不为空的阶段
        # 如果所有机器都忙，时间推进到最早释放的那台机器
        
        stage_idx = -1
        machine_idx = -1
        
        # 简单的离散事件模拟逻辑
        while True:
            # 检查是否有阶段可以调度
            candidates = []
            for s in range(self.num_stages):
                if len(self.queues[s]) > 0:
                    # 找该阶段最早空闲的机器
                    min_time = min(self.machine_available_time[s])
                    m_idx = self.machine_available_time[s].index(min_time)
                    if min_time <= self.current_time:
                        candidates.append((s, m_idx))
            
            if candidates:
                # 优先调度靠后的阶段（简化逻辑，或者由外部循环决定，这里简化为取第一个可行的）
                # 实际上 DRL 应该决定调度哪个阶段的哪个任务，但为了简化，我们固定顺序，DRL 决定“选哪个工件”
                # 这里我们假设 DRL 每次只针对“当前最紧急的决策点”做决策
                stage_idx, machine_idx = candidates[0] 
                break
            
            # 如果没有机器空闲，时间推进
            # 找到整个系统中最早变为空闲的时间
            future_times = [t for stage_times in self.machine_available_time for t in stage_times if t > self.current_time]
            if not future_times:
                # 所有任务都处理完了？
                if self.completed_jobs == self.num_jobs:
                    return self._get_state(), 0, True, {}
                else:
                    # 还有任务在队列但没机器？(理论不应发生，除非逻辑漏洞)
                    pass
            
            if future_times:
                self.current_time = min(future_times)
            else:
                 # 应该结束了
                 current_makespan = max([max(stage) for stage in self.machine_available_time])
                 return self._get_state(), 0, True, {"makespan": current_makespan}

        # ---------------------------------------
        # 根据 Action 选择工件
        # ---------------------------------------
        queue = self.queues[stage_idx]
        selected_job_idx = 0 # 在 queue 中的索引
        
        if len(queue) > 1:
            if action == 0: # SPT
                times = [self.job_times[j, stage_idx] for j in queue]
                selected_job_idx = np.argmin(times)
            elif action == 1: # LPT
                times = [self.job_times[j, stage_idx] for j in queue]
                selected_job_idx = np.argmax(times)
            elif action == 2: # SRPT (剩余总时间)
                rem_times = [np.sum(self.job_times[j, stage_idx:]) for j in queue]
                selected_job_idx = np.argmin(rem_times)
            elif action == 3: # LRPT
                rem_times = [np.sum(self.job_times[j, stage_idx:]) for j in queue]
                selected_job_idx = np.argmax(rem_times)
            elif action == 4: # FIFO
                selected_job_idx = 0
            elif action == 5: # Random
                selected_job_idx = np.random.randint(0, len(queue))
        
        job_id = queue.pop(selected_job_idx)
        
        # ---------------------------------------
        # 执行任务
        # ---------------------------------------
        proc_time = self.job_times[job_id, stage_idx]
        start_time = max(self.current_time, self.machine_available_time[stage_idx][machine_idx])
        
        # 如果是后续阶段，必须等上一阶段完成
        if stage_idx > 0:
            prev_finish = self.job_completion_times[job_id, stage_idx - 1]
            start_time = max(start_time, prev_finish)
            
        finish_time = start_time + proc_time
        
        # 更新状态
        self.machine_available_time[stage_idx][machine_idx] = finish_time
        self.job_completion_times[job_id, stage_idx] = finish_time
        
        # 如果不是最后一个阶段，加入下一阶段队列
        if stage_idx < self.num_stages - 1:
            self.queues[stage_idx + 1].append(job_id)
        else:
            self.completed_jobs += 1
            
        # 计算奖励：负的 Makespan 增量 (鼓励尽快完成)
        # 这里用一种常用的密集奖励：-1 * (当前系统最大完工时间的变化)
        current_makespan = max([max(stage) for stage in self.machine_available_time])
        reward = -0.1 # 每一步的小惩罚
        
        done = (self.completed_jobs == self.num_jobs)
        if done:
            # 完成奖励，与 Makespan 成反比
            reward += 1000.0 / current_makespan
            
        return self._get_state(), reward, done, {"makespan": current_makespan}

# ==========================================
# 2. Neural Network & Agents
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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
        self.memory = deque(maxlen=3000)
        self.tau = 0.1
        
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
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

    def store_transition(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def update(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # Double DQN Logic
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(1).unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values
            
        current_q = self.q_net(states).gather(1, actions)
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_step
        self.soft_update_target_network()

    def soft_update_target_network(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, filename):
        torch.save(self.q_net.state_dict(), filename)

class DQN_Agent(DDTD_Agent):
    def update(self):
        if len(self.memory) < self.batch_size: return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Standard DQN Logic: max_a Q(s', a)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q_values

        current_q = self.q_net(states).gather(1, actions)
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay_step
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
        
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
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

def calculate_makespan_from_sequence(seq, num_stages, job_times):
    return 0 # Placeholder

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
    env.reset()
    env.queues[0] = list(sequence)
    done = False
    final_makespan = 0
    while not done:
        _, _, done, info = env.step(4) # FIFO
        final_makespan = info['makespan']
    return final_makespan

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
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        ms = 0
        while not done:
            action = agent_dqn.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent_dqn.store_transition(state, action, reward, next_state, done)
            agent_dqn.update()
            state = next_state
            ms = info['makespan']
        if ms < best_dqn: best_dqn = ms
    agent_dqn.save_model(f"trained_models/{filename}_DQN.pth")

    # 3. Deep Sarsa (DS)
    agent_ds = DeepSarsa_Agent(env.state_dim, env.action_space.n, max_episodes=num_episodes)
    best_ds = float('inf')
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        ms = 0
        action = agent_ds.select_action(state)
        while not done:
            next_state, reward, done, info = env.step(action)
            ms = info['makespan']
            if not done:
                next_action = agent_ds.select_action(next_state)
                agent_ds.store_transition(state, action, reward, next_state, next_action, done)
                agent_ds.update()
                state = next_state
                action = next_action
            else:
                agent_ds.store_transition(state, action, reward, next_state, 0, done)
                agent_ds.update()
        if ms < best_ds: best_ds = ms
    agent_ds.save_model(f"trained_models/{filename}_DS.pth")

    # 4. DDTD (Ours)
    agent_ddtd = DDTD_Agent(env.state_dim, env.action_space.n, max_episodes=num_episodes)
    best_ddtd = float('inf')
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        ms = 0
        while not done:
            action = agent_ddtd.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent_ddtd.store_transition(state, action, reward, next_state, done)
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
    
    target_instances = {
        "10_5_4", "10_10_5", "10_15_1", "10_20_8",
        "15_5_9", "15_10_2", "15_15_7", "15_20_6",
        "20_5_4", "20_10_5", "20_15_1", "20_20_8",
        "25_5_9", "25_10_2", "25_15_7", "25_20_6",
        "30_5_7", "30_10_1", "30_15_5", "30_20_7",
        "35_5_3", "35_10_4", "35_15_8", "35_20_6"
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