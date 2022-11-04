import wandb

import tempfile
import absl.flags as flags
import ml_collections
import datetime
import wandb

def get_flag_dict():
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict

def default_wandb_config():
    config = ml_collections.ConfigDict()
    config.offline = False # Syncs online or not?
    config.project = 'jaxrl_m' # WandB Project Name
    config.entity = ml_collections.config_dict.FieldReference(None, field_type=str) # Which entity to log as (default: your own user)
    config.exp_prefix = '' # Group name
    config.exp_descriptor = '' # Run name (can be formatted with flags / variant)
    config.unique_identifier = '' # Unique identifier for run (will be automatically generated unless provided)
    return config

def setup_wandb(hyperparam_dict, entity=None, project='jaxrl_m', exp_prefix='', exp_descriptor='', unique_identifier='', offline=False, **additional_init_kwargs):
    """
    Utility for setting up wandb logging (based on Young's simplesac):

    Arguments:
        - hyperparam_dict: dict of hyperparameters for experiment
        - offline: bool, whether to sync online or not
        - project: str, wandb project name
        - entity: str, wandb entity name (default is your user)
        - exp_prefix: str, Group name for wandb
        - exp_descriptor: str, Experiment name for wandb (formatted with FLAGS & hyperparameter_dict)
        - unique_identifier: str, Unique identifier for wandb (default is timestamp)
        - additional_init_kwargs: dict, additional kwargs to pass to wandb.init
    Returns:
        - wandb.run

    """

    if not unique_identifier:
        unique_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_descriptor = exp_descriptor.format(**{**get_flag_dict(), **hyperparam_dict})
    experiment_id = f'{exp_prefix}_{exp_descriptor}_{unique_identifier}'

    wandb_output_dir = tempfile.mkdtemp()

    init_kwargs = dict(
            config=hyperparam_dict,
            project=project,
            entity=entity,
            tags=[exp_prefix],
            group=exp_prefix,
            dir=wandb_output_dir,
            id=experiment_id,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=False,
            ),
            mode='offline' if offline else 'online',
            save_code=True,
        )
    init_kwargs.update(additional_init_kwargs)
    run = wandb.init(**init_kwargs)
    
    wandb.config.update(get_flag_dict())

    wandb_config = dict(
        exp_prefix=exp_prefix,
        exp_descriptor=exp_descriptor,
        experiment_id=experiment_id,
    )
    wandb.config.update(wandb_config)
    return run

