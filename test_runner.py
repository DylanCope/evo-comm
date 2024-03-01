
import jax

import hydra
from omegaconf import OmegaConf

import wandb

from jaxmarl_utils.mappo import MAPPOTrainer


def wandb_login():
    with open('secrets/wandb_api.key', 'r') as key_file:
        key = key_file.read()
        wandb.login(key=key)


@hydra.main(version_base=None,
            config_path="config",
            config_name="mappo_homogenous_rnn_mpe")
def main(config):

    config = OmegaConf.to_container(config)

    wandb_login()

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["MAPPO", "RNN", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )
    rng = jax.random.PRNGKey(config["SEED"])

    @jax.jit
    def train(rng):
        trainer = MAPPOTrainer(config, rng)
        out = trainer.train()
        return out

    with jax.disable_jit(False):
        train(rng)

    
if __name__=="__main__":
    main()
