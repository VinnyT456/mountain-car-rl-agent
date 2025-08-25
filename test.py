import gymnasium as gym
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import time
from model import *
from collections import Counter

plt.ion() 
fig, ax = plt.subplots()
line, = ax.plot([], [], color="blue")
ax.set_title("Live Testing Rewards")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
plt.show(block=False)

def select_action(state, epsilon=0):
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        return q_net(state).argmax().item()
    
def update_plot(rewards):
    line.set_data(range(len(rewards)), rewards)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

# --- Setup ---
env = gym.make('MountainCar-v0')
env.reset(seed=24)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = QNetwork(state_dim, action_dim)
q_net.load_state_dict(torch.load("List Memory/best_mountaincar_model_random_left.pth"))

episodes = 100
average = []

best_average_reward = float('-inf')
best_episode_reward = []
best_episode_actions = []

#Training Loop
for i in range(10):
    rewards, actions = [], []
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

            actions.append(action)

        rewards.append(total_reward)
        update_plot(rewards)

    episode_average_reward = np.mean(rewards)
    if (episode_average_reward > best_average_reward):
        best_average_reward = episode_average_reward
        best_episode_rewards = rewards.copy()
        best_episode_actions = actions.copy()
    print(f'{episode_average_reward:.2f}')
    average.append(episode_average_reward)

    time.sleep(2)


print(f"Average Reward: {np.mean(average):.2f} of {episodes} episodes over 10 trials")

plt.ioff()      
plt.close(fig)      

#Analysis Graphs
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

# 1. Best vs Worst episode
best_idx = np.argmax(best_episode_rewards)
worst_idx = np.argmin(best_episode_rewards)
axes2[0].bar(["Best Episode", "Worst Episode"],
             [best_episode_rewards[best_idx], best_episode_rewards[worst_idx]],
             color=["blue", "red"])
axes2[0].set_ylabel("Total Reward")
axes2[0].set_title("Best vs Worst Episode Reward")

# 2. Action distribution
action_counts = Counter(best_episode_actions)
axes2[1].bar(['Left','Right'], action_counts.values(), color="orange")
axes2[1].set_title("Action Distribution")
axes2[1].set_xlabel("Action")
axes2[1].set_ylabel("Count")

# 3. Reward progression in best trial
axes2[2].plot(best_episode_rewards, color="green")
axes2[2].set_title("Reward Progression (Best Trial)")
axes2[2].set_xlabel("Episode")
axes2[2].set_ylabel("Reward")

plt.show()
