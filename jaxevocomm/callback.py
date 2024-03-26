import wandb


class TrainerCallback:

    def on_train_begin(self, config):
        pass

    def on_train_end(self):
        pass

    def on_iteration_end(self, metric):
        pass


def wandb_try_login():
    with open('secrets/wandb_api.key', 'r') as key_file:
        key = key_file.read()
        wandb.login(key=key)
        print("Logged in to wandb using secrets/wandb_api_key.")


class WandbCallback(TrainerCallback):

    def on_train_begin(self, config):
        self.config = config

        wandb_try_login()

        wandb.init(
            entity=self.config["ENTITY"],
            project=self.config["PROJECT"],
            tags=["MAPPO", "RNN", self.config["ENV_NAME"]],
            config=self.config,
            mode=self.config["WANDB_MODE"],
        )

    def on_train_end(self):
        wandb.finish()

    def on_iteration_end(self, metric):
        wandb.log(
            {
                "returns": metric["returned_episode_returns"][-1, :].mean(),
                "env_step": metric["update_steps"]
                * self.config["NUM_ENVS"]
                * self.config["NUM_STEPS"],
            }
        )
