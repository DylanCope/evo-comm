from typing import List
import wandb

from .callback import TrainerCallback



def wandb_try_login():
    with open('secrets/wandb_api.key', 'r') as key_file:
        key = key_file.read()
        wandb.login(key=key)
        print("Logged in to wandb using secrets/wandb_api_key.")


class WandbCallback(TrainerCallback):

    def __init__(self, tags: List[str] = None):
        self.tags = tags

    def on_train_begin(self, config: dict):
        self.config = config

        wandb_try_login()

        wandb.init(
            entity=self.config["ENTITY"],
            project=self.config["PROJECT"],
            tags=self.tags,
            config=self.config,
            mode=self.config["WANDB_MODE"],
        )

    def on_train_end(self, training_state):
        wandb.finish()

    def on_iteration_end(self, iteration, training_state, metrics):
        wandb.log(metrics)
