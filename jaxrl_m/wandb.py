import wandb

import tempfile
from copy import copy
from socket import gethostname

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

class WandBLogger(object):
    """
    Utility for wandb logging (copied from Young):
    
    wandb_config options:
        - offline: bool, whether to sync online or not
        - project: str, wandb project name
        - entity: str, wandb entity name (default is your user)
        - exp_prefix: str, Group name for wandb
        - exp_descriptor: str, Experiment name for wandb (can be formatted with FLAGS / variant)
        - unique_identifier: str, Unique identifier for wandb (default is timestamp)
    
    variant: dict of hyperparameters to log to wandb

    """
    @staticmethod
    def get_default_config():
        config = ml_collections.ConfigDict()
        config.offline = False # Syncs online or not?
        config.project = 'jaxrl_m' # WandB Project Name
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str) # Which entity to log as (default: your own user)
        config.exp_prefix = '' # Group name
        config.exp_descriptor = '' # Run name (can be formatted with flags / variant)
        config.unique_identifier = '' # Unique identifier for run (will be automatically generated unless provided)
        return config

    def __init__(self, wandb_config, variant=None, wandb_output_dir=None):
        self.config = wandb_config
        if not variant:
            variant = {}

        if not self.config.unique_identifier:
            self.config.unique_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.config.exp_descriptor = self.config.exp_descriptor.format(**{**get_flag_dict(), **variant})
        
        self.config.experiment_id = self.experiment_id = f'{self.config.exp_prefix}_{self.config.exp_descriptor}_{self.config.unique_identifier}'

        print(self.config)

        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp()

        self._variant = copy(variant)

        if 'hostname' not in self._variant:
            self._variant['hostname'] = gethostname()

        self.run = wandb.init(
            config=self._variant,
            project=self.config.project,
            entity=self.config.entity,
            tags=[self.config.exp_prefix],
            group=self.config.exp_prefix,
            dir=wandb_output_dir,
            id=self.config.experiment_id,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode='offline' if self.config.offline else 'online',
            save_code=True,
        )
        wandb.config.update(get_flag_dict())
        wandb.config.update(self.config)

    def log(self, *args, **kwargs):
        wandb.log(*args, **kwargs)

