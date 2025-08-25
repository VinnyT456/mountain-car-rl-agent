import gymnasium as gym
import numpy as np
import torch
from model import *

def select_action(state, epsilon=0):
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        return q_net(state).argmax().item()

# --- Setup ---
env = gym.make('MountainCar-v0')
env.reset(seed=24)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = QNetwork(state_dim, action_dim)
q_net.load_state_dict(torch.load("PER Memory/best_mountaincar_model_per_both.pth"))

episodes = 100
average = []

# --- Training Loop ---
for i in range(10):
    average_reward = 0
    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        total_reward = 0
        done = False

        while not done:
            action = select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

        average_reward += total_reward
    average_reward /= episode
    print(f'{average_reward:.2f}')
    average.append(average_reward)

print(f"Average Reward: {np.mean(average):.2f} of {episodes} episodes over 10 trials")