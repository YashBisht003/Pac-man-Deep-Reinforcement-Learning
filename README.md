# Pac-man-Deep-Reinforcement-Learning
Train an AI agent to master Ms. Pac-Man using Proximal Policy Optimization (PPO) with advanced reward shaping techniques.
Two Training Modes:

Baseline: Standard PPO training with reward clipping
Advanced: Custom reward shaping with strategic ghost mode handling


Advanced Reward Shaping System:

Strategic ghost mode detection and management
Penalties for wasting power pellets
Bonuses for efficient dot collection
Movement encouragement to prevent getting stuck
Multi-ghost eating bonuses
Life management rewards


Optimized Atari Preprocessing Pipeline:

Frame skipping with max pooling
Grayscale conversion
84x84 image resizing
4-frame stacking for temporal information
Normalized pixel values



📋 Requirements
numpy
gymnasium
stable-baselines3[extra]
ale-py
🚀 Installation

Clone the repository:

bashgit clone https://github.com/yourusername/mspacman-rl.git
cd mspacman-rl
