import wandb


class TrainerCallback:

    def on_train_begin(self, config):
        pass

    def on_train_end(self, training_state):
        pass

    def on_iteration_end(self, metric):
        pass



class ChainedCallback(TrainerCallback):

    def __init__(self, *callbacks):
        self.callbacks = callbacks

    def on_train_begin(self, config):
        for cb in self.callbacks:
            cb.on_train_begin(config)

    def on_train_end(self, training_state):
        for cb in self.callbacks:
            cb.on_train_end(training_state)

    def on_iteration_end(self, metric):
        for cb in self.callbacks:
            cb.on_iteration_end(metric)



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

    def on_train_end(self, _):
        wandb.finish()

    def on_iteration_end(self, metric):
        wandb.log(
            {
                "returns": metric["returned_episode_returns"][-1, :].mean(),
                "env_steps": metric["training_iteration"]
                * self.config["NUM_ENVS"]
                * self.config["NUM_STEPS"],
            }
        )
