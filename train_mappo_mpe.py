import hydra
from omegaconf import OmegaConf
import jaxmarl

from jaxevocomm.callback import WandbCallback
from jaxevocomm.mappo import MAPPOTrainer
from jaxevocomm.env.mpe_state_wrapper import MPEWorldStateWrapper
from jaxmarl.wrappers.baselines import MPELogWrapper


@hydra.main(version_base=None,
            config_path="config",
            config_name="mappo_homogenous_rnn_mpe")
def main(config):
    config = OmegaConf.to_container(config)

    env = jaxmarl.make(config["ENV_NAME"],
                       **config["ENV_KWARGS"])
    env = MPEWorldStateWrapper(env)
    env = MPELogWrapper(env)

    cb = WandbCallback()

    trainer = MAPPOTrainer(env, config, cb)
    trainer.run()


if __name__=="__main__":
    main()
