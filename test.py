import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, FrameStackObservation, GrayscaleObservation
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import argparse
import time
import os
import ale_py

class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0


class ToCHW(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape

        if len(old_shape) == 2:
            new_shape = (1, old_shape[0], old_shape[1])
        elif len(old_shape) == 3:
            if old_shape[2] <= 4:
                new_shape = (old_shape[2], old_shape[0], old_shape[1])
            else:
                new_shape = old_shape
        else:
            new_shape = old_shape

        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        if len(obs.shape) == 2:
            return np.expand_dims(obs, axis=0)
        elif len(obs.shape) == 3:
            if obs.shape[2] <= 4:
                return np.transpose(obs, (2, 0, 1))
            else:
                return obs
        else:
            return obs


def make_render_env(seed=0):
    env = gym.make(
        "ALE/MsPacman-v5",
        render_mode="human",  # Enable actual visual rendering
        frameskip=1,
        full_action_space=False
    )
    env.reset(seed=seed)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStackObservation(env, stack_size=4)
    env = NormalizeObservation(env)
    env = ToCHW(env)
    env = ClipRewardEnv(env)
    return DummyVecEnv([lambda: env])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained PPO model')
    args = parser.parse_args()

    # Load the model
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found at {args.model_path}")

    print(f"🎮 Loading model from {args.model_path}")
    env = make_render_env()
    model = PPO.load(args.model_path)
    model.policy.normalize_images = False  # Match training config

    obs = env.reset()
    total_reward = 0
    done = False

    print("🚀 Starting visual test... Close the window or press Ctrl+C to exit.")

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward.sum()

            if done.any():
                print(f"🏁 Episode done. Total Reward: {total_reward}")
                total_reward = 0
                obs = env.reset()

            time.sleep(0.02)  # control frame rate (~50 FPS)
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
