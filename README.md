# 🚗 MountainCar-v0 Reinforcement Learning Agent

This project trains a Double Deep Q-Network (DDQN) to solve the classic MountainCar-v0 environment using a combination of reward shaping, prioritized experience replay (PER), and a velocity-guided heuristic. Despite the sparse reward structure, the agent learns to master the task through momentum-building strategies.

---

🌄 Environment Summary

MountainCar-v0 from OpenAI Gym is a challenging RL environment where:
	•	The car starts between two hills
	•	The goal is to reach the flag on the right hill (position >= 0.5)
	•	The car must build momentum by going back and forth

Feature	Value
Observation	[position, velocity]
Action Space	0 = Push Left, 1 = Do Nothing, 2 = Push Right
Episode Limit	200 steps
Reward	-1 per step until goal is reached

---

## Features:

*	✅ Double Deep Q-Learning (DDQN)
*	🧮 Prioritized Experience Replay (PER)
*	🚀 Velocity-based Custom Reward
*	📊 Epsilon-Greedy Policy with Decay
*	💾 Model Checkpointing

---

## 📁 Project Structure

```
📆 Mountain_Car-RL-Agent
🔼👨‍💼 model.py            # QNetwork architecture (LeakyReLU)
🔼📅 train.py            # Training loop and agent logic
🔼🔧 utils.py            # Custom reward shaping 
🔼🔢 test.py             # Evaluation script to test the trained model
🔼📄 README.md           # Project documentation
🔼📁 requirements.txt    # Python dependencies
```

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Agent

```bash
python train.py
```

This will:

* Train the DQN agent for `10,000` episodes
* Save the best model to `best_black_jack.pth`

### 3. Test the Agent

```bash
python test.py
```

This will:

* Load the trained model
* Run it over `1000` episodes
* Output the average reward of 1000 episodes

---

# 🧠 Agent Architecture

* Architecture: `state_dim -> 32 -> 16 -> 8 -> action_dim`
* Activations: `LeakyReLU`
* Optimizer: `Adam`
* Loss: `MSE`
* Discount Factor: `γ = 0.99`
* Replay Buffers: `PER\Deque\List`
* Epsilon Decay: `ε = 1.0 → 0.005`

---

# 🎯 Reward Shaping

To guide learning through the sparse rewards, a custom reward function was used:

```python
if velocity > 0 and action == 2:
    reward = +1     # Encourages pushing right when moving right
elif velocity < 0 and action == 0:
    reward = +1     # Encourages pushing left when moving left
else:
    reward = -0.01  # Mild penalty for resisting momentum
```

This helps the agent build and maintain momentum, which is critical to reaching the goal.

---

# 🧪 Test Performance

After training, the agent was evaluated over 1000 random-seeded episodes after being trained with 300 episodes of warm up:

## Deque Based Memory

| Warmup Action | Reward without Filtering | Reward with Filtering
| ------------- | ------------------------ | ---------------------
| Left          | \-102.27                 | \-109.16
| Right         | \-106.33                 | \-109.56
| Both          | \-119.60                 | \-104.08

## List Based Memory

| Warmup Action | Reward without Filtering | Reward with Filtering
| ------------- | ------------------------ | ---------------------
| Left          | \-99.88                  | \-103.42
| Right         | \-101.44                 | \-103.59
| Both          | \-105.25                 | \-108.43

## PER Based Memory

| Warmup Action | Reward without Filtering | Reward with Filtering
| ------------- | ------------------------ | ---------------------
| Left          | \-133.01                 | \-122.24
| Right         | \-104.18                 | \-125.14
| Both          | \-112.39                 | \-133.00


The results was typically reached by list based memory which didn't involve remove or changing any memory with the highest reward being -99.88✅

---

# 🗂 Project Structure

```
📁 MountainCar-RL
├── train.py          # Training script with PER and reward shaping
├── test.py           # Testing script for evaluation
├── model.py          # Q-Network architecture
├── utils.py          # Reward shaping and sampling utilities
├── best_model.pth    # Saved best-performing model
├── README.md         # This file
└── requirements.txt  # Dependencies
```

---

# 🔮 Future Ideas

*	Dueling DQN + Double DQN
*	Reward shaping with position-based thresholds
*	Soft target updates
*	Curiosity-driven exploration (RND/ICM)
*	Policy gradient methods (e.g. REINFORCE, A2C)

---

# 📜 License

MIT License

---

# 🙌 Acknowledgments
*	OpenAI Gym for MountainCar-v0
*	PyTorch for the neural network toolkit
*	ChatGPT for collaborative development ideas
