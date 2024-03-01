
import jax

import hydra
from omegaconf import OmegaConf

import wandb
from jaxmarl_utils.callback import WandbCallback

from jaxmarl_utils.mappo import MAPPOTrainer


@hydra.main(version_base=None,
            config_path="config",
            config_name="mappo_homogenous_rnn_mpe")
def main(config):
    config = OmegaConf.to_container(config)
    config['callback_cls'] = WandbCallback
    trainer = MAPPOTrainer(config)
    trainer.run()
    
if __name__=="__main__":
    main()
