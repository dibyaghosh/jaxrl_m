from typing import Dict
import jax
import gym
import numpy as np
from collections import defaultdict
import time

def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
        Wrapper that supplies a jax random key to a function (using keyword `seed`).
        Useful for stochastic policies that require randomness.
    """
    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)
    return wrapped

def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)

def evaluate(policy_fn, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = defaultdict(list)
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        while not done:
            action = policy_fn(observation)
            observation, _, done, info = env.step(action)
            add_to(stats, flatten(info))
        add_to(stats, flatten(info, parent_key='final'))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats

def evaluate_with_trajectories(policy_fn, env: gym.Env,
             num_episodes: int) -> Dict[str, float]:

    trajectories = []
    stats = defaultdict(list)

    for _ in range(num_episodes):
        trajectory = defaultdict(list)
        observation, done = env.reset(), False
        while not done:
            action = policy_fn(observation)
            next_observation, r, done, info = env.step(action)
            transition = dict(observation=observation, next_observation=next_observation,
                action=action, reward=r, done=done, info=info)
            add_to(trajectory, transition)  
            add_to(stats, flatten(info))
            observation = next_observation
        add_to(stats, flatten(info, parent_key='final'))
        trajectories.append(trajectory)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories

class EpisodeMonitor(gym.ActionWrapper):
    """A class that computes episode returns and lengths."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if done:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self, 'get_normalized_score'):
                info['episode']['normalized_return'] = self.get_normalized_score(
                    info['episode']['return']) * 100.0

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()