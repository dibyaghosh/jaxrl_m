import wandb

import tempfile
from copy import copy
from socket import gethostname

import absl.flags as flags
import ml_collections
import datetime
import wandb

class WandBLogger(object):
    """
    Utility for wandb logging (copied from Young):
    
    wandb_config options:
        - offline: bool, whether to sync online or not
        - project: str, wandb project name
        - entity: str, wandb entity name (default is your user)
        - exp_prefix: str, Group name for wandb
        - exp_descriptor: str, Experiment name for wandb
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
        config.exp_descriptor = '' # Run name (doesn't have to be unique)
        config.unique_identifier = '' # Unique identifier for run (will be automatically generated unless provided)
        return config

    def __init__(self, wandb_config, variant=None, wandb_output_dir=None):
        self.config = wandb_config
        if self.config.unique_identifier == '':
            self.config.unique_identifier = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
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
        flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
        for k in flag_dict:
            if isinstance(flag_dict[k], ml_collections.ConfigDict):
                flag_dict[k] = flag_dict[k].to_dict()
        wandb.config.update(flag_dict)

    def log(self, *args, **kwargs):
        wandb.log(*args, **kwargs)

