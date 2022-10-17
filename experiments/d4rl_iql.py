import os
from absl import app, flags
from functools import partial
import numpy as np
import jax
import flax

import tqdm
from custom_agents.mujoco import iql as learner
from custom_agents.mujoco import d4rl_utils

from jaxrl_m.wandb import WandBLogger
from jaxrl_m.evaluation import supply_rng, evaluate

from ml_collections import config_flags
import pickle


FLAGS = flags.FLAGS
flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')

flags.DEFINE_string('save_dir', f'experiment_output/', 'Logging dir.')

flags.DEFINE_integer('seed', np.random.choice(1000000), 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('save_interval', 25000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')

wandb_config = WandBLogger.get_default_config()
wandb_config.update({
    'project': 'd4rl_test',
    'exp_prefix': 'iql_test',
    'exp_descriptor': 'm_iql_{env_name}',
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
    
    env = d4rl_utils.make_env(FLAGS.env_name)
    dataset = d4rl_utils.get_dataset(env)
    def get_normalization(dataset):
        returns = []
        ret = 0
        for r, term in zip(dataset['rewards'], dataset['dones_float']):
            ret += r
            if term:
                returns.append(ret)
                ret = 0
        return (max(returns) - min(returns)) / 1000
    normalizing_factor = get_normalization(dataset)
    print('Scaling rewards down by: ', normalizing_factor)
    dataset = dataset.copy({'rewards': dataset['rewards'] / normalizing_factor})

    print('Config: ', FLAGS.config)
    example_batch = dataset.sample(1)
    agent = learner.create_learner(FLAGS.seed,
                    example_batch['observations'],
                    example_batch['actions'],
                    max_steps=FLAGS.max_steps,
                    **FLAGS.config)

    def add_with_prefix(d, d_to_add, prefix):
        for k, v in d_to_add.items():
            d[f'{prefix}{k}'] = v
    
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       dynamic_ncols=True):

        batch = dataset.sample(FLAGS.batch_size)  
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = dict(iteration=i)
            add_with_prefix(train_metrics, update_info, 'training/')
            wandb_logger.log(train_metrics, step=i)

        if i % FLAGS.eval_interval == 0:
            policy_fn = supply_rng(agent.sample_actions)
            eval_info = evaluate(policy_fn, env, num_episodes=FLAGS.eval_episodes)

            eval_metrics = dict()
            add_with_prefix(eval_metrics, eval_info, 'evaluation/')
            wandb_logger.log(eval_metrics, step=i)

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