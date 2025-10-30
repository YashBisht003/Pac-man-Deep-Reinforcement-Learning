import numpy as np
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStackObservation, GrayscaleObservation
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import os
import argparse
from datetime import datetime

try:
    import ale_py
except ImportError:
    print("Warning: ale_py not found. Install with: pip install ale-py")


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize pixel values to [0, 1] range"""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class ToCHW(gym.ObservationWrapper):
    """Convert observation from HWC or HW to CHW format"""

    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape

        # Handle different input shapes
        if len(old_shape) == 2:  # Single grayscale frame (H, W)
            new_shape = (1, old_shape[0], old_shape[1])
        elif len(old_shape) == 3:  # Either (H, W, C) or stacked frames
            if old_shape[2] <= 4:  # Likely (H, W, C) format
                new_shape = (old_shape[2], old_shape[0], old_shape[1])
            else:  # Already stacked frames in different format
                new_shape = old_shape
        else:  # 4D - already in correct format, don't change
            new_shape = old_shape

        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        if len(obs.shape) == 2:  # Single grayscale (H, W) -> (1, H, W)
            return np.expand_dims(obs, axis=0)
        elif len(obs.shape) == 3:
            if obs.shape[2] <= 4:  # (H, W, C) -> (C, H, W)
                return np.transpose(obs, (2, 0, 1))
            else:  # Already in some other format, return as-is
                return obs
        else:  # 4D or other, return as-is
            return obs


class SmartRewardScaling(gym.RewardWrapper):
    """
    Smart reward scaling that preserves signal ratios but stabilizes training:
    - Scales rewards to reasonable range (0-10)
    - Preserves relative importance (ghost >> dot)
    - Adds strategic bonuses for optimal play
    """

    def __init__(self, env, scale_factor=0.01):
        super().__init__(env)
        self.scale_factor = scale_factor  # Scale rewards down by 100x
        self.last_score = 0
        self.power_pellet_active = False
        self.power_pellet_timer = 0
        self.consecutive_ghosts = 0

    def reset(self, **kwargs):
        self.last_score = 0
        self.power_pellet_active = False
        self.power_pellet_timer = 0
        self.consecutive_ghosts = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Scale base reward to manageable range
        scaled_reward = reward * self.scale_factor

        # Smart reward analysis for strategic bonuses
        if reward > 0:  # Only analyze positive rewards

            # Detect power pellet (usually +50 points)
            if 40 <= reward <= 60:
                self.power_pellet_active = True
                self.power_pellet_timer = 0
                self.consecutive_ghosts = 0
                scaled_reward += 1.0  # Extra bonus for power pellet
                print(f"⚡ Power pellet! Base: {reward}, Scaled: {scaled_reward:.2f}")

            # Track power pellet timer
            if self.power_pellet_active:
                self.power_pellet_timer += 1
                if self.power_pellet_timer > 200:  # Power pellet expires (~8 seconds)
                    self.power_pellet_active = False

            # Ghost eating rewards (200, 400, 800, 1600)
            if reward >= 200:
                if self.power_pellet_active:
                    # Strategic bonuses for optimal ghost hunting
                    combo_multiplier = 1.5 ** self.consecutive_ghosts  # 1.5x per ghost
                    time_multiplier = max(1.0, (200 - self.power_pellet_timer) / 200 + 0.5)

                    scaled_reward *= combo_multiplier * time_multiplier
                    self.consecutive_ghosts += 1

                    print(f"🟡 Ghost eaten! Points: {reward}, "
                          f"Scaled: {scaled_reward:.2f}, Combo: {self.consecutive_ghosts}")

            # Fruit bonus (100-5000 points)
            elif 90 <= reward <= 5000 and reward not in [200, 400, 800, 1600]:
                # Fruit gives good scaled reward + small bonus
                scaled_reward += 0.5
                print(f"🍒 Fruit! Points: {reward}, Scaled: {scaled_reward:.2f}")

            # Small dots/pellets (10-20 points)
            elif reward <= 50:
                # Add tiny survival bonus for consistent dot eating
                if not self.power_pellet_active:
                    scaled_reward += 0.05  # Encourage steady progress

        # Small penalty for losing life (negative reward)
        elif reward < 0:
            scaled_reward = reward * self.scale_factor * 2  # Slightly amplify death penalty

        return obs, scaled_reward, terminated, truncated, info


def make_pacman_env(rank=0, seed=0, render_mode=None, use_reward_shaping=True):
    """Create and wrap Pac-Man environment with strategic preprocessing"""

    def _init():
        env = gym.make(
            "ALE/MsPacman-v5",
            render_mode=render_mode,
            frameskip=1,  # We'll use MaxAndSkipEnv instead
            full_action_space=False
        )

        # Set seeds for reproducibility
        env.reset(seed=seed + rank)

        # Standard Atari preprocessing pipeline
        env = NoopResetEnv(env, noop_max=30)  # Random no-ops at reset
        env = MaxAndSkipEnv(env, skip=4)  # Frame skipping with max pooling
        env = EpisodicLifeEnv(env)  # End episode when life is lost
        env = GrayscaleObservation(env, keep_dim=False)  # Convert to grayscale (H, W)
        env = ResizeObservation(env, shape=(84, 84))  # Resize to 84x84 (H, W)
        env = FrameStackObservation(env, stack_size=4)  # Stack 4 frames -> (H, W, 4)
        env = NormalizeObservation(env)  # Normalize to [0,1]
        env = ToCHW(env)  # Convert (H, W, 4) to (4, H, W)

        # Strategic reward scaling (balances signal preservation with stability)
        if use_reward_shaping:
            env = SmartRewardScaling(env, scale_factor=0.01)  # Scale rewards by 100x

        # NO RAW REWARD CLIPPING, but smart scaling for stability
        # Raw rewards: Ghost=1600, Dot=10, Fruit=100-5000
        # Scaled: Ghost=16, Dot=0.1, Fruit=1-50 + strategic bonuses

        env = Monitor(env)  # Monitor episode stats

        return env

    set_random_seed(seed)
    return _init


def create_directories():
    """Create necessary directories for saving models and logs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"pacman_strategic_training_{timestamp}"

    dirs = {
        'models': os.path.join(base_dir, 'models'),
        'logs': os.path.join(base_dir, 'logs'),
        'eval': os.path.join(base_dir, 'eval')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def main():
    parser = argparse.ArgumentParser(description='Train Strategic PPO on Ms. Pac-Man')
    parser.add_argument('--total-timesteps', type=int, default=10_000_000,
                        help='Total training timesteps (default: 10M for strategic play)')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=256,
                        help='Number of steps per environment per update (increased for Atari)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Minibatch size (increased for better learning)')
    parser.add_argument('--n-epochs', type=int, default=4,
                        help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.995,
                        help='Discount factor (higher for long-term planning)')
    parser.add_argument('--gae-lambda', type=float, default=0.98,
                        help='GAE lambda parameter (higher for strategic play)')
    parser.add_argument('--ent-coef', type=float, default=0.02,
                        help='Entropy coefficient (higher for exploration)')
    parser.add_argument('--vf-coef', type=float, default=1.0,
                        help='Value function coefficient (higher for risk assessment)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to load existing model')
    parser.add_argument('--no-reward-shaping', action='store_true',
                        help='Disable strategic reward shaping')

    args = parser.parse_args()

    print("🎮 Starting STRATEGIC Ms. Pac-Man PPO Training")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Reward clipping: DISABLED ✅")
    print(f"Smart reward scaling: {'DISABLED' if args.no_reward_shaping else 'ENABLED ✅ (0.01x scale)'}")

    # Create directories
    dirs = create_directories()

    # Create vectorized environment
    render_mode = "human" if args.render else None
    use_reward_shaping = not args.no_reward_shaping

    if args.n_envs == 1:
        env = DummyVecEnv([make_pacman_env(0, args.seed, render_mode, use_reward_shaping)])
    else:
        env = SubprocVecEnv([
            make_pacman_env(i, args.seed, render_mode if i == 0 else None, use_reward_shaping)
            for i in range(args.n_envs)
        ])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_pacman_env(100, args.seed + 100, None, use_reward_shaping)])

    # Enhanced PPO hyperparameters for strategic gameplay
    model_kwargs = {
        'learning_rate': args.learning_rate,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,  # Higher discount for long-term planning
        'gae_lambda': args.gae_lambda,  # Better advantage estimation
        'clip_range': 0.1,
        'clip_range_vf': None,
        'normalize_advantage': True,
        'ent_coef': args.ent_coef,  # Higher exploration for strategic play
        'vf_coef': args.vf_coef,  # Stronger value function for risk assessment
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': None,
        'tensorboard_log': dirs['logs'],
        'verbose': 1,
        'seed': args.seed,
        'device': 'auto',
        'policy_kwargs': dict(
            normalize_images=False,  # We normalize manually
            net_arch=[512, 512],  # Larger network for complex strategy
        )
    }

    # Create or load model
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}")
        model = PPO.load(args.load_model, env=env)
        # Update policy kwargs for loaded model
        model.policy.normalize_images = False
    else:
        print("Creating new Strategic PPO model")
        model = PPO("CnnPolicy", env, **model_kwargs)

    # Setup callbacks with more frequent evaluation
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,  # Save every 100k steps
        save_path=dirs['models'],
        name_prefix='strategic_pacman_ppo'
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=dirs['models'],
        log_path=dirs['eval'],
        eval_freq=50_000,  # Evaluate more frequently
        deterministic=True,
        render=False,
        n_eval_episodes=10  # More episodes for better evaluation
    )

    callbacks = [checkpoint_callback, eval_callback]

    try:
        print("\n🚀 Starting strategic training...")
        print("🎯 Expected performance improvements:")
        print("   • 2M steps: ~1,000-2,000 points")
        print("   • 5M steps: ~2,000-4,000 points")
        print("   • 10M steps: ~3,000-6,000+ points")
        print("   • Agent will learn ghost hunting, combos, and strategic positioning!")

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # Save final model
        final_model_path = os.path.join(dirs['models'], 'strategic_pacman_ppo_final')
        model.save(final_model_path)
        print(f"\n✅ Training completed! Final model saved to: {final_model_path}")

        # Test the trained model
        print("\n🎯 Testing trained strategic model...")
        obs = env.reset()
        total_reward = 0
        steps = 0
        episodes = 0
        episode_rewards = []

        for _ in range(5000):  # Test for longer
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward.sum()
            steps += 1

            if done.any():
                episodes += 1
                episode_rewards.append(total_reward)
                print(f"🏆 Episode {episodes} finished: {total_reward:.0f} points in {steps} steps")

                if episodes >= 5:  # Test 5 episodes
                    break

                total_reward = 0
                steps = 0

        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            max_reward = np.max(episode_rewards)
            print(f"\n📊 Test Results:")
            print(f"   Average Score: {avg_reward:.0f} points")
            print(f"   Best Score: {max_reward:.0f} points")
            print(f"   All Scores: {[int(r) for r in episode_rewards]}")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        interrupted_model_path = os.path.join(dirs['models'], 'strategic_pacman_ppo_interrupted')
        model.save(interrupted_model_path)
        print(f"Model saved to: {interrupted_model_path}")

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
