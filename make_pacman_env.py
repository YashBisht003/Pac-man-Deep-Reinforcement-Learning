import numpy as np
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStackObservation, GrayscaleObservation
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv
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


def make_pacman_env(rank=0, seed=0, render_mode=None):
    """Create and wrap Pac-Man environment with proper Atari preprocessing"""

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
        env = ClipRewardEnv(env)  # Clip rewards to {-1, 0, 1}
        env = Monitor(env)  # Monitor episode stats

        return env

    set_random_seed(seed)
    return _init


def create_directories():
    """Create necessary directories for saving models and logs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"pacman_training_{timestamp}"

    dirs = {
        'models': os.path.join(base_dir, 'models'),
        'logs': os.path.join(base_dir, 'logs'),
        'eval': os.path.join(base_dir, 'eval')
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def main():
    parser = argparse.ArgumentParser(description='Train PPO on Ms. Pac-Man')
    parser.add_argument('--total-timesteps', type=int, default=2_000_000,
                        help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=4,
                        help='Number of parallel environments')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=128,
                        help='Number of steps per environment per update')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Minibatch size')
    parser.add_argument('--n-epochs', type=int, default=4,
                        help='Number of epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='GAE lambda parameter')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Path to load existing model')

    args = parser.parse_args()

    print("🎮 Starting Ms. Pac-Man PPO Training")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Parallel environments: {args.n_envs}")
    print(f"Learning rate: {args.learning_rate}")

    # Create directories
    dirs = create_directories()

    # Create vectorized environment
    render_mode = "human" if args.render else None

    if args.n_envs == 1:
        env = DummyVecEnv([make_pacman_env(0, args.seed, render_mode)])
    else:
        env = SubprocVecEnv([
            make_pacman_env(i, args.seed, render_mode if i == 0 else None)
            for i in range(args.n_envs)
        ])

    # Create evaluation environment
    eval_env = DummyVecEnv([make_pacman_env(100, args.seed + 100)])

    # PPO hyperparameters optimized for Atari
    model_kwargs = {
        'learning_rate': args.learning_rate,
        'n_steps': args.n_steps,
        'batch_size': args.batch_size,
        'n_epochs': args.n_epochs,
        'gamma': args.gamma,
        'gae_lambda': args.gae_lambda,
        'clip_range': 0.1,
        'clip_range_vf': None,
        'normalize_advantage': True,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': None,
        'tensorboard_log': dirs['logs'],
        'verbose': 1,
        'seed': args.seed,
        'device': 'auto',
        'policy_kwargs': dict(normalize_images=False)  # We normalize manually
    }

    # Create or load model
    if args.load_model and os.path.exists(args.load_model):
        print(f"Loading model from {args.load_model}")
        model = PPO.load(args.load_model, env=env)
        # Update policy kwargs for loaded model
        model.policy.normalize_images = False
    else:
        print("Creating new PPO model")
        model = PPO("CnnPolicy", env, **model_kwargs)

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=dirs['models'],
        name_prefix='pacman_ppo'
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=dirs['models'],
        log_path=dirs['eval'],
        eval_freq=25_000,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    callbacks = [checkpoint_callback, eval_callback]

    try:
        print("\n🚀 Starting training...")
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            progress_bar=True
        )

        # Save final model
        final_model_path = os.path.join(dirs['models'], 'pacman_ppo_final')
        model.save(final_model_path)
        print(f"\n✅ Training completed! Final model saved to: {final_model_path}")

        # Test the trained model
        print("\n🎯 Testing trained model...")
        obs = env.reset()
        total_reward = 0
        steps = 0

        for _ in range(1000):  # Test for 1000 steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward.sum()
            steps += 1

            if done.any():
                print(f"Episode finished after {steps} steps with total reward: {total_reward}")
                break

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        interrupted_model_path = os.path.join(dirs['models'], 'pacman_ppo_interrupted')
        model.save(interrupted_model_path)
        print(f"Model saved to: {interrupted_model_path}")

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()