import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from model import *
from utils import *

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

#Hyperparameter
gamma = 0.99
lr = 1e-3
epsilon = 1.0
epsilon_decay = 0.997
epsilon_min = 0.005
batch_size = 128
target_update = 5
memory_size = 50000
episodes = 10000

def select_action(state, epsilon, episode):
    if (episode < 300):
        return custom_action(state, episode)
    if random.random() < epsilon:
        return env.action_space.sample()
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        return q_net(state).argmax().item()
    
def sample_per(batch_size, alpha=0.6, beta=0.4):
    current_len = len(memory)
    prios = np.array(priorities[:current_len])

    probs = prios ** alpha
    probs /= probs.sum()

    indices = np.random.choice(current_len, batch_size, p=probs)
    samples = [memory[i] for i in indices]

    weights = ((1 / current_len) * 1 / (probs[indices])) ** beta
    weights /= weights.max()

    weights = torch.tensor(weights, dtype=torch.float32)

    return samples, indices, weights

def train():
    if len(memory) < batch_size:
        return
    #batch, indices, weights = sample_per(batch_size)
    batch = random.sample(memory, batch_size)
    s, a, r, s_, d = zip(*batch)

    s = torch.FloatTensor(np.array(s))
    a = torch.LongTensor(np.array(a)).unsqueeze(1)
    r = torch.FloatTensor(np.array(r)).unsqueeze(1)
    s_ = torch.FloatTensor(np.array(s_))
    d = torch.FloatTensor(np.array(d)).unsqueeze(1)

    q_pred = q_net(s).gather(1, a)

    online_next_actions = q_net(s_).argmax(dim=1, keepdim=True)
    next_q = target_net(s_).gather(1, online_next_actions).detach() 

    q_target = r + gamma * next_q * (1 - d)

    #loss = (weights * (q_pred-q_target).pow(2)).mean()
    loss = nn.MSELoss()(q_pred,q_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    '''td_errors = (q_pred - q_target).abs().detach()
    new_priorities = td_errors + 1e-5

    for idx, prio in zip(indices, new_priorities):
        priorities[idx] = prio.item()'''

#Setup
memory = []
#memory = deque(maxlen=memory_size)
priorities = []
pos = 0

env = gym.make('MountainCar-v0')

env.reset(seed=SEED)
env.action_space.seed(SEED)
env.observation_space.seed(SEED)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = QNetwork(state_dim, action_dim)
target_net = QNetwork(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=lr)

max_reward = -float('inf')
average_reward = [] 
rewards = []

#Training Loop
for episode in range(episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    total_reward = 0
    done = False
    steps = 0

    episode_buffer = []

    while not done:
        action = select_action(state, epsilon, episode)
        next_state, env_reward, terminated, truncated, done = env.step(action)
        done = terminated or truncated

        shaped_reward = env_reward + 0.1 * custom_reward(state, action) + 10 * abs(next_state[0] - state[0])
        shaped_reward -= 0.01  # Time penalty
        if next_state[0] >= 0.5:
            shaped_reward += 1000

        total_reward += env_reward
        #memory.append((state, action, shaped_reward, next_state, done))
        episode_buffer.append((state, action, shaped_reward, next_state, done))
        state = next_state
        train()

    if episode % target_update == 0:
        target_net.load_state_dict(q_net.state_dict())

    average_reward.append(total_reward)
    rewards.append(total_reward)

    for transition in episode_buffer:
        memory.append(transition)

    '''max_priority = 10.0 if total_reward >= -100 else 7.5 if total_reward >= -125 else 5.0 if total_reward >= -150 else 2.5 if total_reward >= -175 else 1.0 if total_reward > -200 else 1e-5
    for transition in episode_buffer: 
        if (len(memory) < memory_size): 
            memory.append(transition)
            priorities.append(max_priority)
        else:
            memory[pos] = transition
            priorities[pos] = max_priority
            pos = (pos + 1) % memory_size'''

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}, Average Reward = {np.mean(average_reward[-100:]):.2f}")

    if (len(average_reward) >= 100):
        if np.mean(average_reward[-100:]) > max_reward and episode > 400:
            max_reward = np.mean(average_reward[-100:])
            torch.save(q_net.state_dict(), 'best_mountaincar_model_per_right.pth')
            print(f"✅ Saved new best model with average reward: {max_reward:.2f}")

print(f"Average Reward of {episode} episodes: {np.mean(np.array(average_reward))}")