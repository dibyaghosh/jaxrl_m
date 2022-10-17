import os
from absl import app, flags
from functools import partial
import numpy as np
import jax
import flax
import tqdm
import gym

from custom_agents.mujoco import sac as learner

from jaxrl_m.wandb import WandBLogger
from jaxrl_m.evaluation import supply_rng, evaluate, flatten, EpisodeMonitor
from jaxrl_m.dataset import ReplayBuffer

from ml_collections import config_flags
import pickle


FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'HalfCheetah-v2', 'Environment name.')

flags.DEFINE_string('save_dir', 'experiment_output/', 'Logging dir.')

flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 25000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_steps', int(1e4), 'Number of initial exploration steps.')

wandb_config = WandBLogger.get_default_config()
wandb_config.update({
    'project': 'd4rl_test',
    'exp_prefix': 'sac_test',
    'exp_descriptor': 'sac_{env_name}',
})

config_flags.DEFINE_config_dict('wandb', wandb_config, lock_config=False)
config_flags.DEFINE_config_dict('config', learner.get_default_config(), lock_config=False)

def main(_):
    # Format exp_descriptor
    FLAG_DICT = {k: getattr(FLAGS, k) for k in FLAGS}
    FLAG_DICT.update(FLAGS.config.to_dict())
    FLAGS.wandb.exp_descriptor = FLAGS.wandb.exp_descriptor.format(**FLAG_DICT)

    # Create wandb logger
    wandb_logger = WandBLogger(
        wandb_config=FLAGS.wandb,
        variant=FLAGS.config.to_dict(),
    )

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, FLAGS.wandb.exp_prefix, wandb_logger.experiment_id)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    env = EpisodeMonitor(gym.make(FLAGS.env_name))
    eval_env = EpisodeMonitor(gym.make(FLAGS.env_name))
    # env = make_env(FLAGS.env_name, FLAGS.seed, None)
    # eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42, None)
    
    example_transition = dict(
        observations=env.observation_space.sample(),
        actions=env.action_space.sample(),
        rewards=0.0,
        masks=1.0,
        next_observations=env.observation_space.sample(),
    )

    replay_buffer = ReplayBuffer.create(example_transition, size=1000000)
    print('Config: ', FLAGS.config)
    agent = learner.create_learner(FLAGS.seed,
                    example_transition['observations'][None],
                    example_transition['actions'][None],
                    max_steps=FLAGS.max_steps,
                    **FLAGS.config)

    def add_with_prefix(d, d_to_add, prefix):
        for k, v in d_to_add.items():
            d[f'{prefix}{k}'] = v
        return d
    
    exploration_metrics = dict()
    obs = env.reset()    
    exploration_rng = jax.random.PRNGKey(0)

    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        if i < FLAGS.start_steps:
            action = env.action_space.sample()
        else:
            exploration_rng, key = jax.random.split(exploration_rng)
            action = agent.sample_actions(obs, seed=key)

        next_obs, reward, done, info = env.step(action)
        mask = float(not done or 'TimeLimit.truncated' in info)

        replay_buffer.add_transition(dict(
            observations=obs,
            actions=action,
            rewards=reward,
            masks=mask,
            next_observations=next_obs,
        ))
        obs = next_obs

        if done:
            add_with_prefix(exploration_metrics, flatten(info), 'exploration/')
            obs = env.reset()

        if replay_buffer.size < FLAGS.start_steps:
            continue

        batch = replay_buffer.sample(FLAGS.batch_size)  
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = dict(iteration=i)
            add_with_prefix(train_metrics, update_info, 'training/')
            wandb_logger.log(train_metrics, step=i)
            wandb_logger.log(exploration_metrics, step=i)
            exploration_metrics = dict()

        if i % FLAGS.eval_interval == 0:
            policy_fn = partial(supply_rng(agent.sample_actions), temperature=0.0)
            eval_info = evaluate(policy_fn, eval_env, num_episodes=FLAGS.eval_episodes)

            eval_metrics = dict()
            add_with_prefix(eval_metrics, eval_info, 'evaluation/')
            wandb_logger.log(eval_metrics, step=i)

            print(train_metrics)
            print(eval_metrics)

        if i % FLAGS.save_interval == 0:
            save_dict = dict(
                agent=flax.serialization.to_state_dict(agent),
                config=FLAGS.config.to_dict()
            )

            fname = os.path.join(FLAGS.save_dir, f'params.pkl')
            print(f'Saving to {fname}')
            with open(fname, "wb") as f:
                pickle.dump(save_dict, f)

if __name__ == '__main__':
    app.run(main)